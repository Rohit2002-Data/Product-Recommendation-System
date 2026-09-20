import io
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import runtime


# ============================================================
# MAKE SURE THE APP IS RUN WITH STREAMLIT
# ============================================================

if not runtime.exists():
    print("Please run this app with:  streamlit run app.py")
    sys.exit(1)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🛍️ Product Recommendation System")

st.write(
    "Enter a new User ID, type the Product IDs this user has rated "
    "and give a rating for each. "
    "The app recommends similar products using cosine similarity."
)


# ============================================================
# UPLOAD TRAINED MODEL
# ============================================================

st.subheader("1. Upload Trained Recommendation Model")

uploaded_file = st.file_uploader(
    "Upload recommendation_model.pkl",
    type=["pkl"],
)

if uploaded_file is None:
    st.info("Please upload your recommendation_model.pkl file.")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model = None

try:
    model = joblib.load(io.BytesIO(uploaded_file.getvalue()))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================
# VALIDATE MODEL
# ============================================================

if not isinstance(model, dict):
    st.error("Uploaded file is not a valid recommendation model.")
    st.stop()

required_keys = ["similarity_matrix", "product_ids", "user_ids"]

missing_keys = [key for key in required_keys if key not in model]

if missing_keys:
    st.error(f"Missing model components: {missing_keys}")
    st.stop()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

similarity_matrix = model["similarity_matrix"]
product_ids = np.asarray(model["product_ids"])
user_ids = np.asarray(model["user_ids"])

st.success("Recommendation model loaded successfully!")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("2. Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Products", len(product_ids))

with col2:
    st.metric("Number of Users", len(user_ids))

with col3:
    st.metric("Similarity Matrix", str(similarity_matrix.shape))


# ============================================================
# HELPERS
# ============================================================

product_ids_string = [str(x) for x in product_ids]
existing_user_ids = {str(x) for x in user_ids}


def normalize_id(value):
    """Make IDs comparable: trim spaces, ignore case, ignore leading zeros
    and a trailing '.0' on numeric IDs (e.g. 0321732944 == 321732944)."""
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text


product_index_map = {}
for i, pid in enumerate(product_ids_string):
    product_index_map.setdefault(normalize_id(pid), i)

# Ratings above this value count as "liked", below it as "disliked".
# It is fixed internally and is not shown in the app.
NEUTRAL_RATING = 3.0


def get_similarity_row(index):
    """Return one row of the similarity matrix as a 1-D float array."""
    if hasattr(similarity_matrix, "getrow"):
        row = similarity_matrix.getrow(index).toarray().flatten()
    else:
        row = np.asarray(similarity_matrix[index]).flatten()
    return np.nan_to_num(row.astype(float))


# ============================================================
# INPUT SECTION (NEW USER + RATED PRODUCTS)
# ============================================================

st.divider()
st.subheader("3. New User Input")

user_id = st.text_input(
    "New User ID",
    value="NEW_USER",
    help="Any ID for the new user. It does not need to be in the dataset.",
)

if user_id.strip() in existing_user_ids:
    st.warning(
        "This User ID already exists in the dataset. "
        "Recommendations below are still based only on the products you select."
    )

st.write(
    "Type the Product IDs this user has rated and a rating for each. "
    "Use the empty row at the bottom of the table to add more products."
)

ratings_input = st.data_editor(
    pd.DataFrame({"Product ID": [""], "Rating": [5.0]}),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="ratings_editor",
    column_config={
        "Product ID": st.column_config.TextColumn(
            "Product ID",
            help="Type any Product ID that exists in the dataset.",
        ),
        "Rating": st.column_config.NumberColumn(
            "Rating",
            min_value=1.0,
            max_value=5.0,
            step=0.5,
        ),
    },
)

with st.expander("Show sample Product IDs from the model"):
    st.write(f"ID data type in the model: {product_ids.dtype}")
    st.write(", ".join(product_ids_string[:20]))

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=5,
)


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if st.button("Get Recommendations", type="primary"):

    # Read the typed Product IDs and ratings
    collected = {}
    not_found = []

    for _, row in ratings_input.iterrows():

        raw_pid = row["Product ID"]
        raw_rating = row["Rating"]

        if pd.isna(raw_pid) or pd.isna(raw_rating):
            continue

        typed_pid = str(raw_pid).strip()

        if typed_pid == "":
            continue

        pid = normalize_id(typed_pid)

        if pid not in product_index_map:
            not_found.append(typed_pid)
            continue

        collected.setdefault(pid, []).append(float(raw_rating))

    # If the same product is typed twice, use the average rating
    user_ratings = {pid: float(np.mean(vals)) for pid, vals in collected.items()}

    if not_found:
        st.warning(
            "These Product IDs were not found in the model and were ignored: "
            + ", ".join(not_found)
            + ". Open 'Show sample Product IDs from the model' above to "
            "check the ID format."
        )

    if not user_ratings:
        st.warning(
            "Please enter at least one valid Product ID with a rating."
        )
        st.stop()

    n_products = len(product_ids_string)

    # Each rating is compared with the midpoint (3):
    #   rating > 3 -> products similar to it get a higher score
    #   rating < 3 -> products similar to it get a lower score
    # score(j)     = sum over rated products i of (rating_i - neutral) * sim(i, j)
    # sim_total(j) = sum over rated products i of sim(i, j)
    score = np.zeros(n_products)
    sim_total = np.zeros(n_products)

    for pid, r in user_ratings.items():
        idx = product_index_map[pid]
        sims = get_similarity_row(idx)
        sims = np.clip(sims, 0, None)  # ignore negative similarities
        score += (r - NEUTRAL_RATING) * sims
        sim_total += sims

    # Predicted rating on the 1-5 scale
    predicted_rating = NEUTRAL_RATING + np.divide(
        score,
        sim_total,
        out=np.zeros_like(score),
        where=sim_total > 0,
    )
    predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

    # Do not recommend products the user already rated
    for pid in user_ratings:
        score[product_index_map[pid]] = -1

    # Rank by score, keep only products with some similarity
    ranked = np.argsort(score)[::-1]
    top_indices = [i for i in ranked if score[i] > 0][:top_n]

    # --------------------------------------------------------
    # DISPLAY INPUT DETAILS
    # --------------------------------------------------------

    st.divider()
    st.subheader("4. Input Details")

    st.metric("User ID", user_id)

    input_df = pd.DataFrame(
        {
            "Rated Product ID": [
                str(product_ids[product_index_map[p]]) for p in user_ratings
            ],
            "Rating": list(user_ratings.values()),
        }
    )

    st.dataframe(input_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # --------------------------------------------------------

    st.divider()
    st.subheader("5. Recommended Products")

    if not top_indices:
        st.warning(
            "No products to recommend. Either none of the products were "
            "rated above 3, or they have no similar products. "
            "Try higher ratings or different Product IDs."
        )
        st.stop()

    result_df = pd.DataFrame(
        [
            {
                "Recommended Product ID": product_ids[i],
                "Score": round(float(score[i]), 4),
                "Predicted Rating": round(float(predicted_rating[i]), 2),
            }
            for i in top_indices
        ]
    )

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.subheader("Recommended Product IDs")

    st.success(
        ", ".join(result_df["Recommended Product ID"].astype(str).tolist())
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Recommendation method: Item-Item Collaborative Filtering "
    "using Cosine Similarity"
)