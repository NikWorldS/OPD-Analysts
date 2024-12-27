import google.generativeai as genai

genai.configure(api_key="AI_API_KEY_HERE")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/custom-laptop-advisor-v3",
    generation_config=generation_config,
    )

response = model.generate_content(["You are an assistant who helps people select the most suitable computer components for people, depending on their daily tasks. Or recommend suitable laptops. For example, if a person is a gamer and wants to run all the new video games, then he needs the appropriate components (video card, RAM). Or if all you need is a laptop for a person who only uses a browser. You should answer \"I dont understand your request for any prompt that not related to PC building or selecting a laptop\"."])

def get_response(message_prompt):
    try:
        return model.generate_content(message_prompt).text
    except Exception as error:
        return f'Похоже, возникла ошибка. [{error}]'

