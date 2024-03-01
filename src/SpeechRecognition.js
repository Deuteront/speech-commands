import React, { useEffect, useState } from 'react';
import * as speechCommands from '@tensorflow-models/speech-commands';
import { Model } from "./model";
import * as tf from '@tensorflow/tfjs';

function SpeechRecognition() {
  const [recognizer, setRecognizer] = useState(null);

  useEffect(() => {
    async function setupRecognizer() {
      await Model();
      const model = await speechCommands.create('BROWSER_FFT');
      await model.ensureModelLoaded();
      model.model = await tf.loadLayersModel('indexeddb://my-model-1');

    }

    setupRecognizer();
  }, []);

  const handleStartListening = async () => {
    if (!recognizer) return;
    recognizer.listen(result => {
      console.log('Recognition result:', result);
    }, { includeSpectrogram: true, probabilityThreshold: 0.75, invokeCallbackOnNoiseAndUnknown: true });
  };

  const handleStopListening = () => {
    if (!recognizer) return;
    recognizer.stopListening();
  };

  return (
    <div>
      <button onClick={ handleStartListening }>Iniciar Reconhecimento de Fala</button>
      <button onClick={ handleStopListening }>Parar Reconhecimento de Fala</button>
    </div>
  );
}

export default SpeechRecognition;