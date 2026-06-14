# app.py
import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Multi-Agent Systems", layout="wide")

st.markdown("""
<style>
.main { background-color: #0b0f19; color: white; }
h1 { color: #FF4B4B; text-align: center; }
.stButton>button { background-color: #FF4B4B; color: white; width: 100%; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Agent Autonomous Research Orchestrator")
st.markdown("<center>Powered by LangGraph, LangChain & Llama-3.1-8B-Instant</center>", unsafe_allow_html=True)
st.divider()

BACKEND_URL = "http://localhost:8000"

query = st.text_input("🔍 Enter a complex research task or problem statement:", 
                     placeholder="e.g., Explain blockchain mining protocols and write a Python script simulator...")

if st.button("🚀 Launch Agentic Pipeline"):
    if not query.strip():
        st.warning("Please type a clear query task first.")
    else:
        # Create clear layout dashboard columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Active Graph Flow Monitor")
            # Display simulated real-time tracking loops
            with st.status("Manager Engine processing...", expanded=True) as status:
                st.write("📥 Booting Supervisor Router Node...")
                st.write("🕵️ Evaluating initial state parameters...")
                st.write("🔁 Deploying worker agents dynamically across memory tracks...")
                
                try:
                    # Fire execution path to FastAPI server backend
                    response = requests.post(f"{BACKEND_URL}/run_research", json={"query": query})
                    
                    if response.status_code == 200:
                        data = response.json()
                        status.update(label="✅ Graph Execution Cycle Complete!", state="complete", expanded=False)
                        
                        st.success(f"Pipeline finished successfully in {data['iterations']} multi-turn iterations.")
                        
                        st.subheader("💡 High-Impact Takeaways")
                        st.info(data["summary"])
                    else:
                        status.update(label="❌ Pipeline Failed", state="error")
                        st.error("Backend returned an execution error processing nodes.")
                        data = None
                except Exception as e:
                    status.update(label="❌ Server Offline", state="error")
                    st.error(f"Could not connect to FastAPI backend: {e}")
                    data = None

        with col2:
            st.subheader("📄 Generated Intelligence Report")
            if data:
                # Output the clean final compiled markdown report text
                st.markdown(data["report"])
            else:
                st.caption("Waiting for pipeline compilation results...")