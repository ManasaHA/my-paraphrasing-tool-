from transformers import T5ForConditionalGeneration, T5Tokenizer

def paraphrase_text(input_text: str):
    # Load pre-trained T5 model and tokenizer
    model_name = "t5-base"  # You can also use 'facebook/bart-large-cnn' for BART model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Encode the input text
    input_ids = tokenizer.encode("paraphrase: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate paraphrase using the model
    output_ids = model.generate(input_ids, num_beams=5, num_return_sequences=1, max_length=256, early_stopping=True)
    
    # Decode the generated paraphrase
    paraphrased_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return paraphrased_text

if __name__ == "__main__":
    input_text = input("Enter text to paraphrase: ")
    paraphrased = paraphrase_text(input_text)
    print("Original Text:", input_text)
    print("Paraphrased Text:", paraphrased)
