import streamlit as st
import asyncio
from deriv_api import DerivAPI

# Initialize session state
if 'response' not in st.session_state:
    st.session_state.response = None

st.title("📡 Deriv WebSocket API - Streamlit Demo")

app_id = st.text_input("Enter your Deriv App ID", "1234")

if st.button("🔌 Connect and Fetch Active Symbols"):
    async def connect_and_fetch():
        try:
            # Connect to Deriv WebSocket API
            api = DerivAPI(endpoint='wss://ws.derivws.com/websockets/v3', app_id=int(app_id))

            # Ping server
            await api.ping({'ping': 1})

            # Get list of active symbols
            active_symbols = await api.active_symbols({'active_symbols': 'brief', 'product_type': 'basic'})

            # Save to session state
            st.session_state.response = active_symbols

            await api.close()

        except Exception as e:
            st.session_state.response = {"error": str(e)}

    asyncio.run(connect_and_fetch())

# Show response
if st.session_state.response:
    st.subheader("🌐 API Response")
    st.json(st.session_state.response)
