#!pip install sentencepiece
#!pip install transformers
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", use_fast=False, src_lang="en_XX")
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Machine Translation</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Conversion from English to Hindi</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">BERT MODEL</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
st.header(" MACHINE TRANSLATION ")
def convert_language(text):
  model_inputs = tokenizer(text, return_tensors="pt")
  translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
  return translation[0] 
  
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    #forced_bos_token_id=tokenizer.lang_code_to_id["ar_AR"]
)

text = st.text_area(" LANGUAGE TRANSLATION")

if st.button(" LANGUAGE TRANSLATION"):
  result=convert_language(text)
  st.success('Model has Translated {}'.format(result))
      
if st.button("About"):
  st.subheader("Developed by Deepak Moud")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;"> Project Deployment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
