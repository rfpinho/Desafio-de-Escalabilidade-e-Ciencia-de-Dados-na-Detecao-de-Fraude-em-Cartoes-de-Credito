import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Carregar e fazer o parse do ficheiro de dados
val data = MLUtils.loadLibSVMFile(sc, "C:\\spark\\creditcard_MLUtils.csv")
// Dividir os dados em conjuntos de treino e de teste (30% para teste)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

// Treinar um modelo de árvores com gradient-boosted
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3
boostingStrategy.treeStrategy.numClasses = 2
boostingStrategy.treeStrategy.maxDepth = 5
// categoricalFeaturesInfo vazio indica que todas as features são contínuas
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val start = System.currentTimeMillis
val model = GradientBoostedTrees.train(trainingData, boostingStrategy)
val tempo_treino = System.currentTimeMillis - start

val start1 = System.currentTimeMillis
// Avaliar o modelo em instâncias de teste e calcular o erro de teste
val labelAndPreds = testData.map { point => 
	val prediction = model.predict(point.features)
    (point.label, prediction)
}
val tempo_classificacao = System.currentTimeMillis - start1

val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification forest model:\n" + model.toDebugString)

// --- Estatisticas (matriz de confusão, precisão)
val metrics = new MulticlassMetrics(labelAndPreds)

// Matriz de confusão
println("Matriz de confusao:")
println(metrics.confusionMatrix)

// Estatisticas gerais
println("Estatisticas sumarias")
println("Precisao = " + metrics.precision)
println("Recall = " + metrics.recall)
println("Tempo de treino = " + tempo_treino)
println("Tempo de classificacao = " + tempo_classificacao)