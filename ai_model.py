import google.generativeai as genai

genai.configure(api_key="AIzaSyDsP3eIyJBkpsevwTgi6VkVK7RoeE64SEw")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="tunedModels/custom_laptop_advisor_v1",
    generation_config=generation_config,
    )

def get_response(message_prompt):
    try:
        return model.generate_content(message_prompt).text
    except Exception as error:
        return f'Похоже, возникла ошибка. [{error}]'