import * as tf from '@tensorflow/tfjs';
import fft from 'fft-js';
import audio1Path from './audios/ola-tasy.mp3';
import audio2Path from './audios/ola-tasy1.mp3';
import audio3Path from './audios/ola-tasy2.mp3';
import audio4Path from './audios/ola-tasy3.mp3';
import audio5Path from './audios/ola-tasy4-falha.mp3';

async function mp3BlobParaAudioData(blob, context) {
  try {
    const url = new URL(blob, window.location.href);
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();

    const audioBuffer = await context.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);

  } catch (error) {
    console.error('Erro ao carregar e decodificar o arquivo MP3:', error);
    return null;
  }
}


async function loadAndPreprocessAudio() {
  const listAudios = [
    { path: audio1Path, value: 1 },
    { path: audio2Path, value: 1 },
    { path: audio3Path, value: 1 },
    { path: audio4Path, value: 1 },
    { path: audio5Path, value: 0 },
  ];

  function normalizeAudio(audioData) {
    const maxAbs = Math.max(...audioData.map(Math.abs));
    return audioData.map(value => value / maxAbs);
  }

  const trainsX = [];
  const trainsY = [];

  for (const audio of listAudios) {
    const context = new window.AudioContext();
    const audioData = await mp3BlobParaAudioData(audio.path, context);
    const normalizedAudioData = normalizeAudio(audioData);
    const spectrogram = createSpectrogram(normalizedAudioData, context.sampleRate);
    debugger
    const spectrogramTensor = tf.tensor2d(spectrogram, [spectrogram.length, spectrogram[0].length]);
    trainsX.push(spectrogramTensor);

    trainsY.push(audio.value);
  }

  return [trainsX, trainsY];
}

function createSpectrogram(audioData) {
  const fftSize = 1024;
  const hopSize = 512;
  const windowFunction = hannWindow;

  const spectrogram = [];
  for (let i = 0; i < audioData.length - fftSize; i += hopSize) {
    const frame = audioData.slice(i, i + fftSize);
    const windowedFrame = applyWindow(frame, windowFunction);
    const spectrum = fft.fft(windowedFrame);
    const magnitude = Array.from(spectrum.map(c => Math.sqrt(c * c)));
    spectrogram.push(magnitude);
  }

  return spectrogram;
}

function applyWindow(frame, windowFunction) {
  return frame.map((value, index) => value * windowFunction(index, frame.length));
}

function hannWindow(length) {
  const window = [];
  for (let i = 0; i < length; i++) {
    window.push(0.5 * (1 - Math.cos((2 * Math.PI * i) / (length - 1))));
  }
  return window;
}

export async function trainAndSaveModel() {
  const [xTrain, yTrain] = await loadAndPreprocessAudio();

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [xTrain[0].shape[0], xTrain[0].shape[1], 1], // Input shape conforme o espectrograma
    kernelSize: [3, 3],
    filters: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  await model.fit(tf.stack(xTrain), tf.tensor(yTrain), {
    batchSize: 32,
    epochs: 10,
    shuffle: true,
    validationSplit: 0.1
  });

  await model.save('downloads://my-model-1');
}