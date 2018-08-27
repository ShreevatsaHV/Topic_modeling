import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.{RandomForestClassifier,RandomForestClassificationModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import org.apache.spark.sql.functions._
import scala.collection.mutable


object Part2 {

  val sc = new SparkContext(new SparkConf().setAppName("Part2"))

  var count = 1
  var rdd :RDD[String]= null

  var bestAirline = ""
  var worstAirline = ""
  def main(args: Array[String]): Unit = {

    if (args.length != 2) {
      println("Usage: WordCount InputDir OutputDir")
    }
    //val sc = new SparkContext(new SparkConf().setAppName("Assignment 3 part 1"))
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val sqlContext = spark.sqlContext
    import spark.implicits._
    var training = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(args(0))
    training = training.filter($"text".isNotNull)

    def udfAirlineNum() = udf[Double, String] { a => val x = a match {
      case "positive" => 5.0;
      case "neutral" => 2.5;
      case "negative" => 1.0;
    }; x;
    }

    val newTrainingDF = training.withColumn("airline_sentiment", udfAirlineNum()($"airline_sentiment"))

    val average = newTrainingDF.groupBy("airline").agg(avg("airline_sentiment").as("airline_sentiment")).sort(desc("airline_sentiment"))
    average.show()
    val top = average.agg(max("airline_sentiment")).head().getDouble(0)
    val least = average.agg(min("airline_sentiment")).head().getDouble(0)

    bestAirline = average.filter($"airline_sentiment" === top).select($"airline").head().getString(0)
    worstAirline = average.filter($"airline_sentiment" === least).select($"airline").head().getString(0)

    //val filteredMax = newTrainingDF.filter($"airline" === bestAirline || $"airline" === worstAirline).select("airline","text")
    val filteredMaBest = newTrainingDF.filter($"airline" === bestAirline).select("text")
    val filteredMaWorst = newTrainingDF.filter($"airline" === worstAirline).select("text")

    topic_Model(filteredMaBest)
    topic_Model(filteredMaWorst)

    rdd.coalesce(1,true).saveAsTextFile(args(1))
  }

  def topic_Model(filteredMax: DataFrame) : Unit ={
    val rdd_filteredMax =
      filteredMax.rdd
        .map(row => {
          val text = row.getString(0)

          (text)
        })
    val stopWordSet = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokenized: RDD[Seq[String]] =
      rdd_filteredMax.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token =>      !stopWordSet.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }
    val numTopics = 5
    val lda = new LDA().setK(numTopics).setMaxIterations(5)

    val ldaModel = lda.run(documents)
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    var output = ""
    output += "The Best Airline is :"+ bestAirline + " and the Worst airline is :" + worstAirline +"\n"
    topicIndices.foreach { case (terms, termWeights) =>
      output += "Topic:" + "\n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output += {vocabArray(term.toInt)}
        output += "\t" + weight + "\n"
      }
      output += "\n\n"

    }
    if(count == 1){
      rdd = sc.parallelize(List(output))
    }else{
      val rdd1 = sc.parallelize(List(output))
      rdd = rdd ++ rdd1
    }

    count = count+1
  }
}
