import * as tf from '@tensorflow/tfjs';
import { loadAndPreprocessAudio } from "./audio/audio";

export async function Model() {

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [44100],
    kernelSize: [3, 3],
    filters: 16,
    activation: 'relu'
  }));
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  const [xTrain, yTrain] = await loadAndPreprocessAudio()
  await model.fit(xTrain, yTrain, {
    batchSize: 32,
    epochs: 10,
    shuffle: true,
    validationSplit: 0.9
  });

  await model.save('downloads://my-model-1');

}

