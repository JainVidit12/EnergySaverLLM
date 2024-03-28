from .AudioProcessor import AudioProcessor
import soundfile as sf

def getCurrentBattery():
    return 0.2


class AgentInput():


    def __init__(
        self,
        model_name : Optional[str] = "openai/whisper-base"
    ):
        self._audioprocessor = AudioProcessor(model_name)

    def getInput(self, input_str) -> str:
        if '.flac' in input_str:
            data, samplerate = sf.read(input_str)
            # if samplerate != 16000:
            #     raise ValueError("flac sample rate should be 16000")
            
            output = self._audioprocessor.processAudio(data, samplerate)[0]
            print(output)
            return output
        else:
            return input_str