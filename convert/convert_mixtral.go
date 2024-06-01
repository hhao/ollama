package convert

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/ollama/ollama/llm"
)

type mixtral struct {
	llama
	NumLocalExperts    uint32 `json:"num_local_experts"`
	NumExpertsPerToken uint32 `json:"num_experts_per_tok"`
}

func (p *mixtral) KV(t *Tokenizer) map[string]any {
	kv := p.llama.KV(t)

	if p.NumLocalExperts > 0 {
		kv["llama.attention.expert_count"] = p.NumLocalExperts
	}

	if p.NumExpertsPerToken > 0 {
		kv["llama.attention.expert_used_count"] = p.NumExpertsPerToken
	}

	return kv
}

func (p *mixtral) Tensors(ts []Tensor) []llm.Tensor {
	oldnew := []string{
		"model.layers", "blk",
		"w1", "ffn_gate_exps",
		"w2", "ffn_down_exps",
		"w3", "ffn_up_exps",
	}

	for i := range p.NumLocalExperts {
		oldnew = append(oldnew, fmt.Sprintf(".block_sparse_moe.experts.%d.", i), ".")
	}

	namer := strings.NewReplacer(oldnew...)

	experts := make(map[string]expert)
	ts = slices.DeleteFunc(ts, func(t Tensor) bool {
		if !strings.Contains(t.Name(), ".block_sparse_moe.experts.") {
			return false
		}

		name := namer.Replace(t.Name())
		experts[name] = append(experts[name], t)
		return true
	})

	var out []llm.Tensor
	for n, e := range experts {
		out = append(out, llm.Tensor{
			Name:     n,
			Kind:     e.Kind(),
			Shape:    e.Shape(),
			WriterTo: e,
		})
	}

	return append(out, p.llama.Tensors(ts)...)
}

type expert []Tensor

func (e expert) Kind() uint32 {
	return e[0].Kind()
}

func (e expert) Shape() []uint64 {
	return e[0].Shape()
}

func (e expert) WriteTo(w io.Writer) (int64, error) {
	for _, t := range e {
		fmt.Println(t.Name())
	}

	return 0, nil
}
