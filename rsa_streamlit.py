"""Streamlit demo of Rational Speech Acts (RSA) model
Jack Morris, jxm3@cornell.edu, 2021-10-12
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from rsa import RSA

def get_core_lexicon() -> pd.DataFrame:
    msgs = pd.Index(['hat', 'glasses'], name='message')
    states = [
        'r1', # guy with glasses but no hat
        'r2'  # guy with glasses and hat
    ]
    lex = pd.DataFrame([
        [0.0, 1.0],
        [1.0, 1.0]], index=msgs, columns=states)
    return lex

def display_reference_game(mod: pd.DataFrame) -> None:
    d = mod.lexicon.copy()
    d['costs'] = mod.costs
    d.loc['prior'] = list(mod.prior) + [""]
    d.loc['alpha'] = [mod.alpha] + [" "] * mod.lexicon.shape[1]
    st.table(d.astype(str))

def main() -> None:
    #
    # Display title and image of guys with glasses and hat
    #
    st.title('Rational Speech Acts (RSA) Model')
    with st.sidebar:
        st.subheader('Referent prior')
        st.markdown('The referent prior can be interpreted as the probability of either referent independent of the conversation.')
        col1, col2 = st.columns(2)
        with col1:
            prior_r1 = st.slider('p(r1)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        with col2:
            prior_r2 = st.slider('p(r2)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        st.subheader('Cost')
        st.markdown('Each message may incur a cost, i.e. it may be much more expensive to say "hat" than "glasses".')
        col1, col2 = st.columns(2)
        with col1:
            cost_r1 = st.slider('Cost("hat")', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        with col2:
            cost_r2 = st.slider('Cost("glasses")', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        st.subheader('Alpha')
        st.markdown('The alpha parameter controls how much pragmatics we see. Larger alphas result in stronger pragmatic inferences.')
        col1, col2 = st.columns(2)
        alpha = st.slider('alpha', min_value=1.0, max_value=5.0, value=1.0, step=1.0)
    col1, col2 = st.columns(2)
    with col1:
        image = cv2.imread('faces.png')
        st.image(image, channels='BGR')
        # st.text('TODO: explanatory text here')
        # st.text('TODO: add buttons for default layouts')
        st.markdown('This is a demonstration of the core ideas behind the Rational Speech Acts model. There are two "referents", shown in the picture. Use the sidebar to adjust parameters and see how they affect the models for the listeners and speaker.')
        # docs.streamlit.io/library/api-reference/widgets/st.select_slider
        # 
        # Display lexicon
        #
        basic_mod = RSA(
            lexicon=get_core_lexicon(),
            prior=[prior_r1, prior_r2],
            costs=[cost_r1, cost_r2],
            alpha=alpha,
        )
        st.subheader('Core lexicon')
        display_reference_game(basic_mod)
    
    with col2:
        # 
        # Display literal listener
        # 
        st.subheader("Literal listener")
        st.latex("P_{Lit}(r|m) = \dfrac{[[m]](r)}{\sum_{r' \in R}{[[m]](r')}}")
        st.table(basic_mod.literal_listener())
        #
        # Display pragmatic speaker
        #
        st.subheader("Pragmatic speaker")
        # TODO consider showing fancy equation here (the one with log and ep)
        st.latex("P_{S}(m|r) = \dfrac{P_{Lit}(r|m)}{\sum_{m' \in M}{P_{Lit}(r|m')}}")
        st.table(basic_mod.speaker())
        #
        # Display pragmatic listener
        #
        st.subheader("Pragmatic listener")
        st.latex("P_{L}(r|m) = \dfrac{P_{S}(m|r)}{\sum_{r' \in R}{P_{S}(m|r')}}")
        st.table(basic_mod.listener())

if __name__ == '__main__': main()