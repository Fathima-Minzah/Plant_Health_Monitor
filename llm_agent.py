from transformers import pipeline

generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def generate_advice(disease_name, category, treatment):

    prompt = f"""
You are an agricultural expert.

Disease: {disease_name}
Category: {category}
Recommended Treatment: {treatment}

Write a short farmer-friendly advisory including:
- Symptoms
- Causes
- Organic solution
- Prevention.

Keep the answer under 120 words.
"""

    output = generator(
        prompt,
        max_length=180,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )

    text = output[0]["generated_text"]

    # Remove prompt part
    cleaned = text.replace(prompt, "").strip()

    return cleaned



