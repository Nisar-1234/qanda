import streamlit as st
import os
import io
import json
import pandas as pd
import docx
import PyPDF2
from openai import OpenAI
import traceback

# PDF text extraction
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# DOCX text extraction
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Excel extraction
def extract_text_from_excel(file):
    df = pd.read_excel(file)
    # Convert DataFrame to text representation
    text = df.to_string(index=False)
    return text

# Initialize OpenAI client
def get_openai_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return None, "API Key is missing. Please enter your OpenAI API key in the sidebar."
        
        client = OpenAI(
            api_key=api_key
        )
        return client, None
    except Exception as e:
        return None, f"Error initializing OpenAI client: {str(e)}"

# Extract content from uploaded file
def extract_content(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_type in ['docx', 'doc']:
            return extract_text_from_docx(uploaded_file)
        elif file_type in ['xlsx', 'xls']:
            return extract_text_from_excel(uploaded_file)
        elif file_type == 'txt':
            return uploaded_file.getvalue().decode('utf-8')
        elif file_type == 'csv':
            df = pd.read_csv(uploaded_file)
            return df.to_string(index=False)
        else:
            return f"Unsupported file type: {file_type}"
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# Generate questions using OpenAI
def generate_questions(content, difficulty_levels, num_questions_per_level, question_types):
    client, error = get_openai_client()
    if error:
        return {"error": error}
    
    try:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Create a structured prompt
        prompt = f"""You are an expert question generator for educational purposes.
Based on the content provided, generate questions for each specified difficulty level.

For each question:
1. Create clear, concise questions
2. Provide the correct answer
3. For multiple choice, include 4 options (A, B, C, D) with one correct answer
4. For each question, include a brief explanation of why the answer is correct

The content to generate questions from is:
{content[:8000]}  # Limiting content length to avoid token limits

Generate {num_questions_per_level} questions for each of these difficulty levels: {', '.join(difficulty_levels)}
Include these question types: {', '.join(question_types)}

Format your response as a valid JSON with the following structure:
{{
  "questions": [
    {{
      "difficulty": "easy|medium|hard",
      "type": "multiple_choice|true_false|short_answer",
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],  # Only for multiple choice
      "correct_answer": "The correct answer",
      "explanation": "Brief explanation of the answer"
    }}
  ]
}}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert educational content creator that specializes in creating high-quality questions from learning materials."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=4000
        )
        
        # Parse the response
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response from OpenAI API", "raw_response": response.choices[0].message.content}
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "error": f"API call failed: {str(e)}",
            "details": error_details
        }

# Main app function
def main():
    st.set_page_config(page_title="Smart Question Generator", layout="wide")
    
    st.title("📚 Smart Question Generator")
    st.write("Upload a document to generate questions of varying difficulty levels")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI Configuration
        st.subheader("OpenAI Settings")
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model = st.selectbox(
            "Model", 
            options=["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo"], 
            index=2
        )
        
        # Update environment variables
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if model:
            os.environ["OPENAI_MODEL"] = model
            
        st.divider()
        
        # Question generation settings
        st.subheader("Question Settings")
        
        # Select difficulty levels
        difficulty_options = ["Easy", "Medium", "Hard"]
        difficulty_levels = st.multiselect(
            "Select difficulty levels", 
            options=difficulty_options,
            default=difficulty_options
        )
        
        # Select question types
        question_type_options = ["Multiple Choice", "True/False", "Short Answer"]
        question_types = st.multiselect(
            "Select question types",
            options=question_type_options,
            default=["Multiple Choice"]
        )
        
        # Number of questions per difficulty level
        num_questions = st.slider("Number of questions per difficulty level", 1, 10, 3)
    
    # Main panel
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "xlsx", "xls", "csv"])
        
        if uploaded_file is not None:
            with st.spinner("Extracting content..."):
                content = extract_content(uploaded_file)
                
                # Show content preview
                st.subheader("Content Preview")
                st.text_area("Extracted Text", content[:1000] + ("..." if len(content) > 1000 else ""), height=200)
                
                # Generate questions button
                if st.button("Generate Questions"):
                    if not difficulty_levels:
                        st.error("Please select at least one difficulty level")
                    elif not question_types:
                        st.error("Please select at least one question type")
                    else:
                        with st.spinner("Generating questions..."):
                            # Map UI-friendly names to backend names
                            difficulty_mapping = {
                                "Easy": "easy",
                                "Medium": "medium",
                                "Hard": "hard"
                            }
                            
                            question_type_mapping = {
                                "Multiple Choice": "multiple_choice",
                                "True/False": "true_false",
                                "Short Answer": "short_answer"
                            }
                            
                            backend_difficulty_levels = [difficulty_mapping[d] for d in difficulty_levels]
                            backend_question_types = [question_type_mapping[t] for t in question_types]
                            
                            result = generate_questions(
                                content=content,
                                difficulty_levels=backend_difficulty_levels,
                                num_questions_per_level=num_questions,
                                question_types=backend_question_types
                            )
                            
                            # Display results in second column
                            with col2:
                                st.header("Generated Questions")
                                
                                # Check if there was an error
                                if isinstance(result, dict) and 'error' in result:
                                    st.error(result['error'])
                                    if 'details' in result:
                                        with st.expander("Error Details"):
                                            st.code(result['details'])
                                    if 'raw_response' in result:
                                        with st.expander("Raw Response"):
                                            st.code(result['raw_response'])
                                else:
                                    # Create tabs for each difficulty level
                                    if 'questions' in result:
                                        # Group questions by difficulty
                                        questions_by_difficulty = {}
                                        for q in result['questions']:
                                            difficulty = q['difficulty']
                                            if difficulty not in questions_by_difficulty:
                                                questions_by_difficulty[difficulty] = []
                                            questions_by_difficulty[difficulty].append(q)
                                        
                                        # Create tabs
                                        tabs = st.tabs([d.capitalize() for d in questions_by_difficulty.keys()])
                                        
                                        # Display questions in each tab
                                        for i, (difficulty, questions) in enumerate(questions_by_difficulty.items()):
                                            with tabs[i]:
                                                for j, q in enumerate(questions):
                                                    with st.expander(f"Question {j+1}: {q['question'][:60]}{'...' if len(q['question']) > 60 else ''}"):
                                                        st.write(f"**Question:** {q['question']}")
                                                        st.write(f"**Type:** {q['type'].replace('_', ' ').title()}")
                                                        
                                                        # Display options for multiple choice
                                                        if q['type'] == 'multiple_choice' and 'options' in q:
                                                            st.write("**Options:**")
                                                            for k, option in enumerate(q['options']):
                                                                option_label = chr(65 + k)  # A, B, C, D...
                                                                st.write(f"{option_label}. {option}")
                                                        
                                                        st.write(f"**Correct Answer:** {q['correct_answer']}")
                                                        st.write(f"**Explanation:** {q['explanation']}")
                                        
                                        # Add download buttons
                                        st.subheader("Export Options")
                                        col1, col2 = st.columns(2)
                                        
                                        # JSON export
                                        with col1:
                                            json_str = json.dumps(result, indent=4)
                                            st.download_button(
                                                label="Download as JSON",
                                                data=json_str,
                                                file_name="generated_questions.json",
                                                mime="application/json"
                                            )
                                        
                                        # Text export (for easy copying)
                                        with col2:
                                            text_output = ""
                                            for diff, questions in questions_by_difficulty.items():
                                                text_output += f"# {diff.upper()} QUESTIONS\n\n"
                                                for i, q in enumerate(questions):
                                                    text_output += f"Question {i+1}: {q['question']}\n"
                                                    if q['type'] == 'multiple_choice' and 'options' in q:
                                                        for j, opt in enumerate(q['options']):
                                                            text_output += f"{chr(65+j)}. {opt}\n"
                                                    text_output += f"Answer: {q['correct_answer']}\n"
                                                    text_output += f"Explanation: {q['explanation']}\n\n"
                                            
                                            st.download_button(
                                                label="Download as Text",
                                                data=text_output,
                                                file_name="generated_questions.txt",
                                                mime="text/plain"
                                            )
                                    else:
                                        st.error("No questions were generated. Please try again.")

if __name__ == "__main__":
    main()