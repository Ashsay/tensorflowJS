import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

window.onload = async () => {

  const heights = [150, 160, 170];
  const weights = [40, 50, 60];

  // 可视化数据
  tfvis.render.scatterplot(
    { name: "身高体重训练数据" },
    { values: heights.map((x, i)=>({x, y: weights[i]})) },
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70]
    }
  )

  //创建连续模型
  const model = tf.sequential();
  //units: 一个神经元  inputShape: 特征数量
  model.add(tf.layers.dense({units:1, inputShape:[1]}))
  model.compile({ 
    //设置损失函数
    loss: tf.losses.meanSquaredError,
    //设置优化器  学习率
    optimizer: tf.train.sgd(0.1) 
  });

  //归一化数据
  const inputs = tf.tensor(heights).sub(150).div(20);
  const labels = tf.tensor(weights).sub(40).div(20);

  await model.fit(inputs, labels, {
    //所需要学习的样本数据量
    batchSize: 3,
    //迭代次数
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      { name: "训练过程" },
      ['loss']
    )
  });
  
  //预测结果  归一化输入数据
  const output = model.predict(tf.tensor([180]).sub(150).div(20));
  output.print();
  //反归一化输出结果
  console.log(output.mul(20).add(40).dataSync()[0]);
  document.getElementById("result").innerHTML = output.mul(20).add(40).dataSync()[0] + "kg";
  
}