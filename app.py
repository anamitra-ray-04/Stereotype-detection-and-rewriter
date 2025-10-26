import streamlit as st
import os
from dotenv import load_dotenv

# Import our custom components
from detector import StereotypeDetector

def main():
    # Page config
    st.set_page_config(page_title="Stereotype Detection System", page_icon="", layout="wide")
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please set GROQ_API_KEY in your .env file")
        st.stop()
    
    # Initialize system
    try:
        detector = StereotypeDetector(api_key)
    except Exception as e:
        st.error(f"Error initializing systems: {str(e)}")
        st.stop()
    
    # Set up the main interface
    st.title("Stereotype Detection and Rewriting System")
    st.write("Enter text to analyze for stereotypes and get bias-free alternatives.")

    # Text input section
    text_input = st.text_area("Enter your text:", height=150)

    if st.button("Analyze and Rewrite"):
        if not text_input:
            st.error("Please enter some text to analyze.")
            return

        try:
            # Analyze text for stereotypes
            with st.spinner("Analyzing text..."):
                result = detector.detect_stereotype(text_input)
            
            # Display results
            st.subheader("Analysis Results")
            
            # Show LLM Results
            st.markdown("### LLM Analysis")
            if result["success"]:
                if result["has_stereotype"]:
                    # Show stereotype detection results
                    st.warning(
                        f"⚠️ Stereotype detected (Severity Score: {result['severity_score']:.2f})"
                    )
                    
                    # Show stereotype type if available
                    if result.get("stereotype_type"):
                        st.error(f"Type: {result['stereotype_type']}")
                    
                    # Show detailed analysis
                    st.markdown("#### Detailed Analysis")
                    st.write(result["explanation"])
                else:
                    st.success("✅ No stereotypes detected by LLM")
            else:
                st.error(f"Error in LLM analysis: {result['error']}")
            
            # Show Logistic Regression Results
            st.markdown("### Logistic Regression Analysis")
            log_reg_result = detector.predict_with_logreg(text_input)
            if log_reg_result["success"]:
                # Create two columns for probabilities
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "No Stereotype Probability", 
                        f"{log_reg_result['probabilities']['no_stereotype']:.2%}"
                    )
                
                with col2:
                    st.metric(
                        "Stereotype Probability", 
                        f"{log_reg_result['probabilities']['stereotype']:.2%}"
                    )
                
                if log_reg_result["has_stereotype"]:
                    st.warning(f"⚠️ Stereotype detected (Confidence: {log_reg_result['confidence']:.2%})")
                else:
                    st.success(f"✅ No stereotype detected (Confidence: {log_reg_result['confidence']:.2%})")
            else:
                st.error(f"Error in logistic regression analysis: {log_reg_result.get('error')}")
            
            # Show BERT Results (optional)
            st.markdown("### BERT Analysis")
            try:
                bert_result = detector.predict_with_bert(text_input)
            except Exception as e:
                bert_result = {"success": False, "error": str(e)}

            if bert_result.get("success"):
                # Create two columns for probabilities when binary
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "BERT No Stereotype Probability",
                        f"{bert_result['probabilities'].get('no_stereotype', 0.0):.2%}"
                    )

                with col2:
                    st.metric(
                        "BERT Stereotype Probability",
                        f"{bert_result['probabilities'].get('stereotype', 0.0):.2%}"
                    )

                if bert_result["has_stereotype"]:
                    st.warning(f"⚠️ BERT detected stereotype (Confidence: {bert_result['confidence']:.2%})")
                else:
                    st.success(f"✅ BERT: No stereotype detected (Confidence: {bert_result['confidence']:.2%})")
            else:
                # If BERT model isn't loaded or produced an error, show info but don't break
                st.info("BERT model not available or error in BERT analysis.")
                if bert_result.get('error'):
                    st.error(f"BERT error: {bert_result['error']}")

            # Generate and show bias-free version if stereotype detected by LLM
            if result["success"] and result["has_stereotype"]:
                with st.spinner("Generating bias-free version..."):
                    rewritten_text = detector.rewrite_text(text_input)
                    if not rewritten_text.startswith("Error"):
                        st.subheader("Suggested Bias-free Version")
                        st.success(rewritten_text)
                    else:
                        st.error(rewritten_text)
                        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()