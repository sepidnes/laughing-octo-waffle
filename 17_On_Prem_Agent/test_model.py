from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    print("Model and tokenizer loaded successfully!")

    # Test the model with a simple prompt
    prompt = "Hello, how are you?"
    print(f"\nTesting with prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\nModel response: {response}")

if __name__ == "__main__":
    main() 