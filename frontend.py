import streamlit as st
import sys

# Import from meld_based
from meld_based import search, get_dataset_info

# ======================================================
# 1. Page configuration
# ======================================================
st.set_page_config(
    page_title="🎬 dAIrectors - Semantic Footage Search",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. Custom CSS Styling
# ======================================================
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .header-container {
        text-align: center;
        padding: 2rem 0;
        color: white;
    }
    
    .header-container h1 {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .search-container {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .result-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .result-rank {
        background: #667eea;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .result-movie {
        font-weight: bold;
        font-size: 1.1rem;
        color: #333;
    }
    
    .result-time {
        color: #666;
        font-size: 0.9rem;
        font-family: monospace;
    }
    
    .confidence-badge {
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .result-text {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        font-style: italic;
        color: #333;
        border-left: 3px solid #667eea;
        margin-top: 1rem;
    }
    
    .sidebar-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-info h4 {
        color: #aebbf2;
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
        font-weight: bold;
    }

    .sidebar-info p {
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .no-results {
        text-align: center;
        padding: 3rem;
        background: #fff3cd;
        border-radius: 0.5rem;
        color: #856404;
    }
    
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ======================================================
# 4. Main UI
# ======================================================

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1>🎬 dAIrectors</h1>
            <p>Semantic Footage Search Engine</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        
        top_k_results = st.slider(
            "Results to return",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Number of results to display"
        )
        
        min_confidence = st.slider(
            "Minimum confidence (%)",
            min_value=20,
            max_value=80,
            value=40,
            step=5,
            help="Filter results by minimum confidence score"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Dataset Information")
        
        # Get dataset info
        try:
            info = get_dataset_info()
            
            st.markdown(f"""
                <div class="sidebar-info">
                    <h4>📁 Total Clips</h4>
                    <p>{info['total_clips']:,}</p>
                    <h4>🎬 Movies</h4>
                    <p>{info['unique_movies']}</p>
                    <h4>🧠 Embedding Model</h4>
                    <p>{info['model_name']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading dataset info: {e}")
        
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "hesitant reaction",
            "emotional dialogue",
            "awkward pause",
            "dramatic scene",
            "confused character",
            "nervous moment"
        ]
        st.markdown("\n".join([f"• {q}" for q in example_queries]))
    
    # Main search area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔎 Enter your search query",
            placeholder="e.g., 'hesitant reaction', 'emotional dialogue', 'awkward pause'",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True, type="primary")
    
    # Display results
    if search_button and query:
        if len(query.strip()) < 3:
            st.warning("⚠️ Please enter a query with at least 3 characters.")
        else:
            with st.spinner("🔍 Searching through subtitles..."):
                results = search(query, top_k=top_k_results, min_confidence=min_confidence)
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            if results:
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Top Match</h3>
                            <div class="value">{results[0]['confidence']}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    avg_confidence = sum(r['confidence'] for r in results) / len(results)
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Avg Confidence</h3>
                            <div class="value">{avg_confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Results Found</h3>
                            <div class="value">{len(results)}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                
                st.markdown("### 🎞️ Results (Ranked by Confidence)")
                
                # Display results
                for idx, result in enumerate(results, 1):

                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">
                                <div style="display: flex; gap: 1rem; align-items: center;">
                                    <div class="result-rank">#{idx}</div>
                                    <div>
                                        <div class="result-movie">🎬 {result['movie']}</div>
                                        <div class="result-time">⏱️ {result['start_time']} → {result['end_time']}</div>
                                    </div>
                                </div>
                                <div class="confidence-badge">
                                    {result['confidence']}%
                                </div>
                            </div>
                            <div class="result-text">
                                💬 {result['text']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                    <div class="no-results">
                        <h3>❌ No Results Found</h3>
                        <p>Try lowering the minimum confidence threshold or using different keywords.</p>
                        <p>Current threshold: <strong>{min_confidence}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🎞️ dAIrectors | Semantic Footage Search Engine</p>
            <p>Powered by Sentence Transformers & FAISS Vector Search</p>
        </div>
    """, unsafe_allow_html=True)

# ======================================================
# 5. Run Application
# ======================================================

if __name__ == "__main__":
    main()