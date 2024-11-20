import traceback
import streamlit as st
from openai import OpenAI

def get_prompt(prompt):
    return prompt
def get_fireworks_code_completions(example, **kwargs):
    prompt = example
    prompt = get_prompt(prompt)
    
    kwargs = kwargs.copy()
    api_key = kwargs.pop('api_key')

    Tries = 10
    for _ in range(Tries):
        try:
            with OpenAI(
                base_url = "https://api.fireworks.ai/inference/v1",
                api_key = api_key
            ) as client:
                params_dict = {
                    'prompt': prompt,
                    **kwargs,
                }
                response = client.completions.create(**params_dict)
                
                # Debug prints
                print("Full Response:", response)
                response_text = response.choices[0].text
                print("Response Text:", response_text)

                return response_text
        
        except Exception as e:
            traceback.print_exc()
            print(f"Unknown error: {e}\n For model: {kwargs['model']}, tokens: {len(prompt)/4}")
            raise e

all_kwargs = {
    'api_key': 'fw_3ZX7cBYPHUuwe9gBBFxwDF2G', 
    'model': 'accounts/sourcegraph/models/deepseek-coder-v2-lite-base', 
    'temperature': 0.4,
    'max_tokens': 256,
    'stop': [
        "\n\n",
        "\n\r\n",
        "<BLANKLINE>",
        "<|eos_token|>"
    ]

}



st.title("Javascript code generator")
user_input = st.text_area("write your prompt",height=300,label_visibility="visible")
if st.button("Autocomplete"):
    if user_input:
        try:
            result = get_fireworks_code_completions(user_input, **all_kwargs)
            st.write("Processed string:", user_input+result)
        except Exception as e:
            st.error(f"Error processing the string: {e}")
    else:
        st.write("Please enter a string to process.")