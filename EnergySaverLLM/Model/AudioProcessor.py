from transformers import WhisperProcessor, WhisperForConditionalGeneration

from typing import Optional



class AudioProcessor():
    
    def __init__(
        self,
        model_name : Optional[str] = "openai/whisper-small"
    ):
        self._processor = WhisperProcessor.from_pretrained(model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self._model.config.forced_decoder_ids = None


    def processAudio(self, data, sampling_rate) -> str:
        input_features = self._processor(data, sampling_rate=sampling_rate, return_tensors="pt").input_features 
        predicted_ids = self._model.generate(input_features)
        transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription