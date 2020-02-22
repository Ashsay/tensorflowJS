/**
 * @description: 欠拟合现象
 */
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import { getData } from '../xor/data';

window.onload = async () => {

  const data = getData(200);
  console.log(data);

  tfvis.render.scatterplot(
    { name: '训练数据' },
    {
      values: [
        data.filter(p=>p.label === 1),
        data.filter(p=>p.label === 0)
      ]
    }
  )

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    inputShape: [2]
  }))
  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
  })

  const inputs = tf.tensor(data.map(p=>[p.x, p.y]));
  const labels = tf.tensor(data.map(p=>p.label));

  await model.fit(inputs, labels, {
    validationSplit: .2,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练效果' },
      ['loss', 'val_loss'],
      { callbacks: ['onEpochEnd'] }
    )
  })

}