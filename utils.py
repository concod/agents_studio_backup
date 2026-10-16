# from langchain.embeddings import HuggingFaceEmbeddings
import os
import time
import httpx



async def get_llm_response(prompt: str, return_all_candidates: bool = False, api_key: str = None) -> str | list[str]:
    """Makes a synchronous call to the Gemini API with a given prompt.
    
    Args:
        prompt: The input prompt for the API.
        return_all_candidates: If True, returns all candidate responses as a list.
                              If False, returns the first candidate's text as a string.
    
    Returns:
        A string (if return_all_candidates=False) or a list of strings (if True).
    """
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    if not api_key:
        error_msg = "GEMINI_API_KEY environment variable is not set"
        print(error_msg)
        raise ValueError(error_msg)

    max_attempts = 3
    retry_delays = [4, 8, 10]  # Delays in seconds for each retry attempt

    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{endpoint}?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": prompt}]
                        }],
                        "generationConfig": {
                            "maxOutputTokens": 500,  # Increased for potentially longer outputs
                            "temperature": 0.7
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Log the full response for debugging
                print(f"API Response: {result}")

                # Extract generated text(s)
                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError("Empty response from Gemini API: no candidates returned")

                generated_texts = [
                    candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    for candidate in candidates
                ]
                generated_texts = [text for text in generated_texts if text]  # Filter out empty strings

                if not generated_texts:
                    raise ValueError("Empty response from Gemini API: no text generated")

                # Return based on return_all_candidates flag
                if return_all_candidates:
                    return generated_texts
                return generated_texts[0]

        except Exception as e:
            if attempt < max_attempts - 1:
                delay = retry_delays[attempt]
                print(f"Retrying ({attempt + 1}/{max_attempts}) after {delay}s due to error: {str(e)}")
                time.sleep(delay)
                continue
            else:
                error_msg = f"Failed to get response from Gemini API after {max_attempts} retries: {str(e)}"
                print(error_msg)
                raise