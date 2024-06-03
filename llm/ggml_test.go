package llm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"text/template"
)

func TestKVChatTemplate(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "templates.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	expect := map[string]string{
		"91aa728ae59c8e30443382f44f48d594b5f7afe91a449c0e5ed0e920a71d60a2": "chatml",
		"58c1a1f04baa7adaeaba1f90267c3d57d9396f0c1f0129bdde9e76d3c6784af7": "chatml",
		"3ce2fe9753f9a77a601f55dfcab7cd46fbc91324aae0f78ca38d194a4eaef8d5": "chatml",
		"a805e50fed68938a076b07e2e602639611b50b1ced0e50f11eb92f1ba25be4dc": "chatml",
		"153280e3ff55d19da1398bdb3914ee2a51b80429bfaedde11d7d216c39db80f3": "chatml",
		"793202280c0910ab26bc6eb57c8212c2417324c6d8c4efd0d10d59092ab3e3eb": "chatml",
		"af9c0233881b083b52ff773580215222b5440ac3d0beeeca99b76329b048f8db": "chatml",
		"e6f6e5c192657d1631c2cdf89cd8402518e7bf3f44dce4c4a7d9be74bfaa946a": "chatml",
		"f42d04ffb2e487fa9c501ab886afe8fce0822f9874a9ca4284ba57e2a686a56a": "chatml",
		"f02c534193010c4bd1e40fe3dd501147fa46cebadb0769d97217a2c785028783": "chatml",
		"faf1f8b675beeabd4ddbb89b93bb03f94f097b3bb1ff5dfa8e99e70674a9cdbb": "chatml",
		"ea2472f29dbd93c3eb696795f6403228fcc4b775518869c771f83fbae5c11dbf": "llama2-chat",
		"e8d5b5a546d3f19df16a5320c614fab05bdc4edd50e850f1afd947b503ea41cf": "llama2-chat",
		"ba03a121d097859c7b5b9cd03af99aafe95275210d2876f642ad9929a150f122": "llama3-instruct",
		"26a59556925c987317ce5291811ba3b7f32ec4c647c400c6cc7e3a9993007ba7": "mistral-instruct",
		"7e995b379ec01747807246483647cd99030abf331653f1119e16d7ac041a3495": "mistral-instruct",
		"ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6": "gemma-instruct",
		"66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362": "phi",
		"bf61473d7119388a9093c4e55e4d83651e68e1cfa985a82093a0d1a2ac114514": "phi",
		"605841622bcb6572225608f09ec2afcdfb9fb50844ef52715a12689f7b9278b1": "phi",
		"268b6082ceb7176dc6ed80557a2f7837f9f0339592fbee677d405a553af15f88": "phi",
		"b0e53f5e7bae6fdb93013abeceef343e8841cce87def2601e7269a53439b5740": "phi",
		"901b10d47195fadb5692137ecd4b2733730be9d32fde6c6eae23e5153712525e": "openchat",
		"b8e5e922a7a4ab5df61a904ef54639a27022fd84eb6b6434f09503afeca60eb4": "solar-instruct",
		"993c0af704bad76aeba4263a5318f7873a06420a8eb7d40b72a49c39599c0a66": "alpaca",
		"8aeba567270fa9a8d5372caf4b04affea947c4421d96adc01e3ff94021b0ae8e": "chatqa",
		"cd3636885332999fc1931dc2a0070f767bc5b2af643ab05c6dc48806add23b3f": "chatqa",
		"2c575663fbb6c03a6af2cb20f5e8b3b764af5d66c6465d4067ff5c8b25ee3d0b": "falcon-instruct",
		"f6084d6ec1cb3238db36169ad4bb1ecec9ef3d18a55e80347c8c7c4d6e5507a3": "falcon-instruct",
		"6de66c5b4eb23df882a50a73aa7759464b45dbc05495538ffbd91df35fa6c823": "starcoder2-instruct",
		"75678d463085865163b4c6dbc28f7178738d992b53124ad3612304f48a983b02": "codellama-70b-instruct",
		"e84ee90c8a81a15d98c58d6b541fc5a5f207507e88f5d8338b64040e40430161": "granite-instruct",
		"230527c8a608cf3a28953748af58780929fe77bd472e0044b1ad427bff8c956a": "magicoder",
		"071cfb2d4c797ad44dffad0ced15636e9ab349a23c5bba3c81e187947159cab5": "alfred",
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var ss map[string]string
		if err := json.Unmarshal(scanner.Bytes(), &ss); err != nil {
			t.Fatal(err)
		}

		for k, v := range ss {
			t.Run(k, func(t *testing.T) {
				kv := KV{"tokenizer.chat_template": v}
				actual := kv.ChatTemplate()

				if actual != expect[k] {
					t.Errorf("expected %q, got %q", expect[k], actual)
				}

				r, err := NamedTemplate(actual)
				if err != nil {
					t.Fatal(err)
				}

				var b bytes.Buffer
				if _, err := io.Copy(&b, r); err != nil {
					t.Fatal(err)
				}

				tmpl, err := template.New(actual).Parse(b.String())
				if err != nil {
					t.Fatal(err)
				}

				if tmpl.Tree.Root.String() == "" {
					t.Errorf("empty %s template", expect[k])
				}
			})
		}
	}
}
