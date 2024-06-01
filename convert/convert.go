package convert

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/llm"
)

type Parameters struct {
	Architectures []string `json:"architectures"`
	VocabSize     uint32   `json:"vocab_size"`
}

func (Parameters) KV(t *Tokenizer) map[string]any {
	kv := map[string]any{
		"general.file_type":         uint32(1),
		"tokenizer.ggml.pre":        t.Pre,
		"tokenizer.ggml.model":      t.Vocabulary.Model,
		"tokenizer.ggml.tokens":     t.Vocabulary.Tokens,
		"tokenizer.ggml.scores":     t.Vocabulary.Scores,
		"tokenizer.ggml.token_type": t.Vocabulary.Types,
	}

	if t.Template != "" {
		kv["tokenizer.chat_template"] = t.Template
	}

	for _, sv := range t.SpecialVocabulary {
		kv[fmt.Sprintf("tokenizer.ggml.%s_token_id", sv.Key())] = uint32(sv.ID)
		kv[fmt.Sprintf("tokenizer.ggml.add_%s_token", sv.Key())] = sv.AddToken
	}

	return kv
}

func (Parameters) SpecialTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

type Converter interface {
	KV(*Tokenizer) map[string]any
	Tensors([]Tensor) []llm.Tensor
	SpecialTypes() []string
	tensorName(string) string
}

func Convert(d string, ws io.WriteSeeker) error {
	f, err := os.Open(filepath.Join(d, "config.json"))
	if err != nil {
		return err
	}
	defer f.Close()

	var p Parameters
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return err
	}

	if len(p.Architectures) < 1 {
		return errors.New("unknown architecture")
	}

	var c Converter
	switch p.Architectures[0] {
	case "LlamaForCausalLM", "MistralForCausalLM":
		c = &llama{}
	case "MixtralForCausalLM":
		c = &mixtral{}
	case "GemmaForCausalLM":
		c = &gemma{}
	default:
		return errors.New("unsupported architecture")
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return err
	}

	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return err
	}

	t, err := parseTokenizer(d, c.SpecialTypes())
	if err != nil {
		return err
	}

	if vocabSize := int(p.VocabSize); vocabSize > len(t.Vocabulary.Tokens) {
		slog.Warn("vocabulary is smaller than expected, padding with dummy tokens", "expect", p.VocabSize, "actual", len(t.Vocabulary.Tokens))
		for i := range vocabSize - len(t.Vocabulary.Tokens) {
			t.Vocabulary.Tokens = append(t.Vocabulary.Tokens, fmt.Sprintf("<dummy%05d>", i))
			t.Vocabulary.Scores = append(t.Vocabulary.Scores, -1)
			t.Vocabulary.Types = append(t.Vocabulary.Types, tokenTypeUserDefined)
		}
	}

	ts, err := parseTensors(d)
	if err != nil {
		return err
	}

	return llm.WriteGGUF(ws, c.KV(t), c.Tensors(ts))
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}
