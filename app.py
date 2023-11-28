from faster_whisper import WhisperModel
import streamlit as st
from tempfile import NamedTemporaryFile

# model_size = "large-v2"
# filepath = "/Users/hanimedialab/Downloads/"
# filename = "test.mp3"
# file_head = filename.split(".")[0]
# mp3_file = filepath + filename
# TXT 파일에 저장
def write_txt(filepath, text):
    with open(filepath, 'a+', encoding='utf-8') as f:
        f.write(text)

def main():
    st.image("https://cdn.pixabay.com/photo/2017/01/31/13/50/headphones-2024215_1280.png", width=150)
    st.title("Hani Script Extractor")
    st.subheader("Convert MP3 to TXT")
    st.markdown("오픈AI의 오픈소스 인공지능 STT(Speech-to-Text) 모델인 [Whisper](https://github.com/openai/whisper)를 활용했습니다. ")

    # whisper model 선택
    whisper_model = st.selectbox("모델을 선택해주세요.(base나 small을 권장합니다. medium과 large는 스크립트 추출 속도가 느려지거나 오류가 날 수 있습니다.)", ('tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2'))
    st.write("모델 : ", whisper_model) 
    st.divider()
    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    # 파일 업로드
    mp3_file = st.file_uploader("MP3 파일을 올려주세요.", type=["mp3"])

    if mp3_file is not None:
    #        progress_text = "Operation in progress. Please wait."
    #        my_bar = st.progress(0, text=progress_text)
    #        for percent_complete in range(100):
    #            time.sleep(0.1)
    #            my_bar.progress(percent_complete + 1, text=progress_text)
    #        time.sleep(3)
        try:
            with st.spinner("스크립트를 추출하고 있습니다. 실시간 변환 중이라 생각보다 시간이 오래 걸릴 수 있어요. 잠시만 기다려주세요..."):
                with NamedTemporaryFile(suffix="mp3", delete=False) as tmp_file:
                    tmp_file.write(mp3_file.getvalue())
                    file_path = tmp_file.name

               # Extract Script
                model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
                segments, info = model.transcribe(mp3_file, beam_size=5)
                st.write("Detected language '%s' with probability %f" % (info.language, info.language_probability))
                with NamedTemporaryFile(suffix="txt", delete=False) as tmp_text:
                    for segment in segments:
                        txt = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                        txt_file = tmp_text.name + '.txt'
                        write_txt(txt_file, txt + '\n')
                        st.write(txt)
                        print(txt)
                    if txt_file:
                        st.success("스크립트 추출 완료!")
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            full_txt = f.read()
                            st.write(full_txt)
        except Exception as e:
            st.error("오류가 발생했습니다. 😥")
            st.write("웹페이지를 새로고침한 후 모델을 'base'나 'small'로 지정하고 재시도해보세요.")

        # 다운로드 링크 생성
        file_name = '-'.join(mp3_file.name.split(".")[:-1]) + ".txt"
        # file_bytes = txt_file.encode()
        st.download_button(label="Download Script", data=full_txt, file_name=file_name)

if __name__ == "__main__":
    main()

# 출처 : https://github.com/SYSTRAN/faster-whisper
