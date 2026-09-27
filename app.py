import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import os
from ml.predict import predict_packaging_compatibility

st.set_page_config(page_title="AI Biscuit Packaging", page_icon="🍪", layout="wide")

# Initialize session state for history
if 'assessment_history' not in st.session_state:
    st.session_state.assessment_history = []

def load_metadata():
    try:
        with open('models/model_metadata.json', 'r') as f:
            return json.load(f)
    except Exception:
        return {}

metadata = load_metadata()

def load_registry():
    try:
        return pd.read_csv('processed_data/packaging_pairing_registry.csv')
    except Exception:
        return None
# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "🔬 Compatibility Assessment",
    "📊 Model Insights",
    "📦 Packaging Materials",
    "📋 Assessment History",
    "ℹ️ About the Model"
])

def render_dashboard():
    st.title("🍪 AI Biscuit Packaging Compatibility")
    st.markdown("### ML-powered packaging decision-support dashboard")

    cv_acc = metadata.get('cv_accuracy')
    cv_acc_str = f"{cv_acc*100:.2f}%" if cv_acc is not None else "Unavailable"

    registry_df = load_registry()
    if registry_df is not None:
        num_products = str(registry_df['product_id'].nunique())
        num_materials = str(registry_df['packaging_material'].nunique())
    else:
        num_products = "Error"
        num_materials = "Error"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ML Model", metadata.get('model_type', 'Random Forest'))
    col2.metric("Grouped CV Accuracy", cv_acc_str)
    col3.metric("Products Evaluated", num_products)
    col4.metric("Packaging Materials", num_materials)

    st.markdown("---")
    st.header("Ready for Assessment")
    st.markdown("> Enter biscuit and packaging characteristics to generate an ML-based compatibility assessment.")

    st.markdown("### How it works")
    st.code("""Biscuit Characteristics
        ↓
Packaging Properties
        ↓
Evidence Quality
        ↓
ML Preprocessing
        ↓
Random Forest Model
        ↓
Compatibility Prediction""", language="text")

def render_assessment():
    st.title("🔬 Compatibility Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🍪 Biscuit Characteristics")
        initial_moisture = st.number_input("Initial Moisture Content (g/100g)", min_value=0.0, max_value=100.0, value=3.5, step=0.1)
        final_moisture = st.number_input("Final Observed Moisture Content (g/100g)", min_value=0.0, max_value=100.0, value=5.5, step=0.1)

        moisture_gain = final_moisture - initial_moisture
        st.markdown(f"""
        ```text
        Initial Moisture      {initial_moisture:.1f}
                               ↓
        Final Observed Moisture {final_moisture:.1f}
                               ↓
        Moisture Gain         {moisture_gain:.2f} g/100g
        ```
        """)

    with col2:
        st.subheader("📦 Packaging Characteristics")
        packaging_material = st.selectbox("Material", ["LDPE", "HDPE", "BOPP", "PET", "Metallized PET", "EVOH Laminate"])

        material_info = {
            "LDPE": "Flexible polymer packaging material. Barrier properties: based on the values supplied for this assessment.",
            "HDPE": "High-density polyethylene. Barrier properties: based on the values supplied for this assessment.",
            "BOPP": "Biaxially Oriented Polypropylene. Barrier properties: based on the values supplied for this assessment.",
            "PET": "Polyethylene terephthalate. Barrier properties: based on the values supplied for this assessment.",
            "Metallized PET": "PET with a thin metal coating. Barrier properties: based on the values supplied for this assessment.",
            "EVOH Laminate": "Ethylene vinyl alcohol laminate. Barrier properties: based on the values supplied for this assessment."
        }
        st.info(f"**{packaging_material}**\n\n{material_info[packaging_material]}")

        thickness_micron = st.number_input("Thickness (microns)", min_value=1.0, max_value=500.0, value=30.0, step=1.0)
        otr = st.number_input("Oxygen Transmission Rate (OTR) cm³/m²/day", min_value=0.0, value=150.0, step=10.0)
        wvtr = st.number_input("Water Vapor Transmission Rate (WVTR) g/m²/day", min_value=0.0, value=5.0, step=0.5)

        st.subheader("📑 Evidence Quality")
        evidence_quality = st.selectbox(
            "Literature Evidence Quality",
            [1.0, 0.8],
            format_func=lambda x: "High (DOI-linked source)" if x == 1.0 else "Standard (Representative)"
        )

    st.markdown("---")

    if st.button("🚀 Run Compatibility Assessment", type="primary", use_container_width=True):


        user_input = {
            'initial_moisture': initial_moisture,
            'final_moisture': final_moisture,
            'thickness_micron': thickness_micron,
            'wvtr': wvtr,
            'otr': otr,
            'packaging_material': packaging_material,
            'evidence_quality': evidence_quality
        }

        try:
            with st.spinner("Running ML compatibility assessment..."):
                result = predict_packaging_compatibility(user_input)

            st.success("Assessment completed.")

            st.markdown("---")
            st.header("COMPATIBILITY ASSESSMENT")

            rec = result['recommendation']
            prob = result['probability']

            if rec == 'Recommended':
                st.success(f"### 🟢 {rec.upper()}\n**Model Probability: {prob*100:.1f}%**")
            elif rec == 'Conditionally Recommended':
                st.warning(f"### 🟡 {rec.upper()}\n**Model Probability: {prob*100:.1f}%**")
            else:
                st.error(f"### 🔴 {rec.upper()}\n**Model Probability: {prob*100:.1f}%**")

            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Moisture Gain", f"{moisture_gain:.2f} g/100g")
            col_res2.metric("Packaging", packaging_material)

            ev_label = "High" if evidence_quality == 1.0 else "Standard"
            col_res3.metric("Evidence", ev_label)

            st.markdown("### Assessment Summary")
            st.markdown(f"> The trained ML model classified the submitted biscuit-packaging configuration as **{rec}** based on patterns learned from the available training dataset.")

            if 'class_probabilities' in result:
                st.markdown("### Model Prediction Distribution")
                for cls_name, cls_prob in result['class_probabilities'].items():
                    col_p1, col_p2, col_p3 = st.columns([3, 6, 1])
                    with col_p1:
                        st.markdown(f"**{cls_name}**")
                    with col_p2:
                        st.progress(cls_prob)
                    with col_p3:
                        st.markdown(f"{cls_prob*100:.1f}%")

            st.markdown("### 🔎 Prediction Factors")
            st.markdown(f"""
            ```text
            Biscuit
            Initial moisture       {initial_moisture:.2f} g/100g
            Final observed moisture       {final_moisture:.2f} g/100g
            Moisture gain          {moisture_gain:.2f} g/100g

            Packaging
            Material               {packaging_material}
            Thickness              {thickness_micron} µm
            WVTR                   {wvtr}
            OTR                    {otr}

            Evidence
            Quality                {ev_label}
            ```
            > **The model evaluates these characteristics together to classify the packaging configuration.**
            """)

            # Save to history
            record = {
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Initial moisture": initial_moisture,
                "Final moisture": final_moisture,
                "Moisture Gain": moisture_gain,
                "Material": packaging_material,
                "Thickness": thickness_micron,
                "WVTR": wvtr,
                "OTR": otr,
                "Evidence quality": ev_label,
                "Prediction": rec,
                "Probability": f"{prob*100:.1f}%"
            }
            if 'class_probabilities' in result:
                for cls_name, cls_prob in result['class_probabilities'].items():
                    record[f"Prob: {cls_name}"] = f"{cls_prob*100:.1f}%"
            st.session_state.assessment_history.append(record)

            st.markdown("---")
            report_content = f"""Assessment ID: {datetime.now().strftime('%Y%m%d%H%M%S')}
Date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Biscuit characteristics:
- Initial moisture: {initial_moisture:.2f} g/100g
- Final observed moisture: {final_moisture:.2f} g/100g
- Moisture gain: {moisture_gain:.2f} g/100g

Packaging characteristics:
- Material: {packaging_material}
- Thickness: {thickness_micron} µm
- WVTR: {wvtr}
- OTR: {otr}

ML prediction: {rec}
Model Probability: {prob*100:.1f}%
"""
            if 'class_probabilities' in result:
                report_content += "\nClass Probabilities:\n"
                for cls_name, cls_prob in result['class_probabilities'].items():
                    report_content += f"- {cls_name}: {cls_prob*100:.1f}%\n"

            report_content += f"""
Model information:
- Type: {metadata.get('model_type', 'Random Forest')}
- CV Accuracy: {f"{metadata.get('cv_accuracy')*100:.2f}%" if metadata.get('cv_accuracy') is not None else "Unavailable"}

Technical note: {metadata.get('training_limitations', 'Model trained on deterministic rules.')}

Limitation: The Random Forest model performs the final prediction. Training labels were generated by the deterministic compatibility framework; therefore model probability/accuracy represents agreement with rule-generated labels. It is NOT an experimentally measured probability of real-world packaging compatibility. The project does not currently have independent experimental shelf-life ground truth.
"""
            st.download_button("📄 Download Assessment Report", data=report_content, file_name=f"assessment_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    with st.expander("ℹ️ Model Scope & Technical Notes"):
        st.info("The trained Random Forest model generated this assessment from the submitted biscuit and packaging characteristics. Training labels were generated from deterministic rules, and these metrics indicate agreement with the existing decision logic rather than independent experimental validation.")

def render_insights():
    st.title("📊 Model Insights")
    st.header("Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Algorithm:** {metadata.get('model_type', 'Random Forest')}")
        st.markdown(f"**Training samples:** {metadata.get('training_sample_count', 126)}")
        st.markdown(f"**Holdout samples:** {metadata.get('testing_sample_count', 36)}")
    with col2:
        st.markdown(f"**Features:** {len(metadata.get('feature_names', []))}")
        st.markdown(f"**Classes:** {len(metadata.get('target_classes', []))}")

    st.header("Validation")
    st.subheader("5-Fold Product-Grouped Cross-Validation")
    cv_acc_val = metadata.get('cv_accuracy')
    st.markdown(f"### **{f'{cv_acc_val*100:.2f}%' if cv_acc_val is not None else 'Unavailable'}**")
    st.markdown("> Products were grouped during validation so that observations from the same biscuit product were not split across training and validation folds.")

    st.header("Model Validation: Confusion Matrix")
    if os.path.exists('reports/confusion_matrix.png'):
        st.image('reports/confusion_matrix.png')
    else:
        st.info("Confusion matrix image not found.")

def render_materials():
    st.title("📦 Packaging Materials")
    st.markdown("Reference dataset values for the packaging materials.")

    registry_df = load_registry()
    if registry_df is not None:
        try:
            materials_df = registry_df[['packaging_material', 'thickness_micron', 'wvtr', 'otr']].drop_duplicates().dropna(subset=['packaging_material']).copy()
            materials_df = materials_df.rename(columns={
                'packaging_material': 'Material',
                'thickness_micron': 'Thickness',
                'wvtr': 'WVTR',
                'otr': 'OTR'
            })
            materials_df = materials_df[materials_df['Material'].isin(["LDPE", "HDPE", "BOPP", "PET", "Metallized PET", "EVOH Laminate"])]

            # Sort explicitly by the original order if desired, or alphabetically
            # Here we sort alphabetically for simplicity
            materials_df = materials_df.sort_values(by='Material').reset_index(drop=True)
            st.table(materials_df)
        except Exception as e:
            st.error(f"Application error: Could not process material values from registry. Details: {str(e)}")
    else:
        st.error("Application error: Could not load packaging materials from the registry.")

def render_history():
    st.title("📋 Assessment History (Session-based)")

    if len(st.session_state.assessment_history) == 0:
        st.info("No assessments run in this session.")
    else:
        df = pd.DataFrame(st.session_state.assessment_history)
        st.dataframe(df[["Timestamp", "Material", "Moisture Gain", "Prediction", "Probability"]] if "Timestamp" in df.columns else df)

        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Assessment History CSV",
            data=csv,
            file_name="assessment_history.csv",
            mime="text/csv",
        )

def render_about():
    st.title("ℹ️ About the Model")
    st.markdown("### AI Biscuit Packaging Compatibility Prediction System")
    st.markdown("ML Decision-Support Prototype | Random Forest | v1.0")
    st.markdown("**Model validation:** Product-grouped 5-fold CV")
    st.markdown(metadata.get('training_limitations', ''))

if page == "🏠 Dashboard":
    render_dashboard()
elif page == "🔬 Compatibility Assessment":
    render_assessment()
elif page == "📊 Model Insights":
    render_insights()
elif page == "📦 Packaging Materials":
    render_materials()
elif page == "📋 Assessment History":
    render_history()
elif page == "ℹ️ About the Model":
    render_about()
