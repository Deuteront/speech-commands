import * as jsfeat from 'jsfeat';
import * as tf from '@tensorflow/tfjs';

export async function loadAndPreprocessAudio() {
  const listAudios = [
    { path: 'ola-tasy.mp3', value: 1 },
    { path: 'ola-tasy1.mp3', value: 1 },
    { path: 'ola-tasy2.mp3', value: 1 },
    { path: 'ola-tasy3.mp3', value: 1 },
    { path: 'ola-tasy4-falha.mp3', value: 0 },
  ];
  const trainsX = [];
  const trainsY = [];

  const context = new window.AudioContext();

  for (const audio of listAudios) {
    trainsY.push(audio.value)
    const response = await fetch('./audios/' + audio.path);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await context.decodeAudioData(arrayBuffer);

    // espectrograma
    const audioData = audioBuffer.getChannelData(0);
    const spectrogram = new jsfeat.matrix_t(1, 1, jsfeat.F32C1_t);
    jsfeat.audio.spectrogram(audioData, spectrogram);

    trainsX.push(tf.tensor2d(spectrogram.data, [spectrogram .rows, spectrogram .cols]));

  }
  return [trainsX, trainsY];
}
