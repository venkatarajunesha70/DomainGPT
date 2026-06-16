"""
DomainGPT – Streamlit UI
Run with: streamlit run apps/ui/streamlit_app.py
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="DomainGPT",
    page_icon="🧠",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN / REGISTER SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 DomainGPT")
    st.caption("Enterprise RAG + LoRA Chatbot")
    st.divider()

    if not st.session_state.token:
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True):
                resp = requests.post(
                    f"{API_BASE}/auth/login",
                    json={"email": email, "password": password},
                )
                if resp.ok:
                    st.session_state.token = resp.json()["access_token"]
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Login failed"))

        with tab_reg:
            reg_email = st.text_input("Email", key="reg_email")
            reg_username = st.text_input("Username", key="reg_username")
            reg_pass = st.text_input("Password", type="password", key="reg_pass")
            reg_tenant = st.text_input("Tenant ID", value="default", key="reg_tenant")
            if st.button("Register", use_container_width=True):
                resp = requests.post(
                    f"{API_BASE}/auth/register",
                    json={
                        "email": reg_email,
                        "username": reg_username,
                        "password": reg_pass,
                        "tenant_id": reg_tenant,
                    },
                )
                if resp.ok:
                    st.session_state.token = resp.json()["access_token"]
                    st.success("Registered and logged in!")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Registration failed"))
    else:
        st.success("✅ Logged in")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # ── Document upload ───────────────────────────────────────────────
        st.subheader("📎 Upload Document")
        uploaded = st.file_uploader(
            "PDF, DOCX, TXT, PNG, JPG",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        )
        if uploaded and st.button("Upload & Index", use_container_width=True):
            resp = requests.post(
                f"{API_BASE}/documents/upload",
                headers=auth_headers(),
                files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
            )
            if resp.ok:
                data = resp.json()
                st.success(f"✅ {data['message']}")
                st.caption(f"Document ID: `{data['id']}`")
            else:
                st.error(resp.json().get("detail", "Upload failed"))

        st.divider()

        # ── New conversation ───────────────────────────────────────────────
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
st.title("💬 Chat with your Documents")

if not st.session_state.token:
    st.info("👈 Please log in from the sidebar to start chatting.")
    st.stop()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Sources"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**[{c['index']}]** {c['filename']} "
                        f"{'— p.' + str(c['page_num']) if c.get('page_num') else ''}"
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_BASE}/chat/",
                headers=auth_headers(),
                json={
                    "question": prompt,
                    "conversation_id": st.session_state.conversation_id,
                },
            )

        if resp.ok:
            data = resp.json()
            st.session_state.conversation_id = data["conversation_id"]
            answer = data["answer"]
            citations = data.get("citations", [])
            rewritten = data.get("rewritten_query", "")

            st.markdown(answer)

            if citations:
                with st.expander("📚 Sources"):
                    for c in citations:
                        st.markdown(
                            f"**[{c['index']}]** `{c['filename']}` "
                            f"{'— p.' + str(c['page_num']) if c.get('page_num') else ''}"
                        )

            if rewritten and rewritten != prompt:
                st.caption(f"🔍 Search query: *{rewritten}*")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })
        else:
            err = resp.json().get("detail", "Error from API")
            st.error(f"❌ {err}")
