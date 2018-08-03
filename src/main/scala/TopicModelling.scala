import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{ Tokenizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.clustering.{LDA}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import scala.collection.mutable.HashMap


object TopicModelling {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    if (args.length != 2) {
      println("I/p and O/p filepath needed")
    }
    Logger.getLogger("labAssignment").setLevel(Level.OFF)
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    val tweetData = spark.read.option("header","true")
      .csv(args(0))
    val cols = Array("text")
    val filteredTweetData = tweetData.na.drop(cols)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")


    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover))

    val preProcessedData = pipeline.fit(filteredTweetData)
    val tweetPreProcessedData = preProcessedData.transform(filteredTweetData)
    val tweetTransformedData = tweetPreProcessedData.withColumn("airline_sentiment", when(lower(col("airline_sentiment")).equalTo("positive"), 5.0).otherwise(when(lower(col("airline_sentiment")).equalTo("negative"), 1.0).otherwise(2.5) ))
    val avgRatingData = tweetTransformedData.groupBy("airline")
      .agg(
        avg("airline_sentiment").as("avg_rating")
      )

    val avgRatingAirlines = avgRatingData.select("airline").orderBy(desc("avg_rating"))

    val airlineList = avgRatingAirlines.select("airline").map(r => r.getString(0)).collect.toArray

    val bestAirline = airlineList(0)

    val worstAirline = airlineList(airlineList.length-1)

    val bestAirlineData = tweetTransformedData.filter(tweetPreProcessedData("airline") === bestAirline)

    val textData= bestAirlineData.map(_.getAs[Seq[String]]("filtered"))

    val textDataRDD = textData.rdd

    val termCounts: Array[(String, Long)] =
      textDataRDD.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val vocabArray: Array[String] =
      termCounts.map(_._1)

    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    val documents: RDD[(Long, Vector)] =
      textDataRDD.zipWithIndex.map { case (tokens, id) =>
        val counts = new HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    val numTopics = 10
    val lda = new LDA().setK(numTopics).setMaxIterations(100)

    val ldaModel = lda.run(documents)
    val printContent = new StringBuilder()

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 20)
	  printContent.append("Topics for Best Airline: "+bestAirline+"\n")
    topicIndices.foreach { case (terms, termWeights) =>
      printContent.append("TOPIC:"+"\n")
      terms.zip(termWeights).foreach { case (term, weight) =>
        printContent.append(s"${vocabArray(term.toInt)}\t$weight")
        printContent.append("\n")
      }
      printContent.append("\n")
    }
	
	val worstAirlineData = tweetTransformedData.filter(tweetPreProcessedData("airline") === worstAirline)
	val textData1= worstAirlineData.map(_.getAs[Seq[String]]("filtered"))
	val textDataRDD1 = textData1.rdd
	val termCounts1: Array[(String, Long)] =
		textDataRDD1.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
	val vocabArray1: Array[String] =
		termCounts1.map(_._1)
	val vocab1: Map[String, Int] = vocabArray1.zipWithIndex.toMap
	val documents1: RDD[(Long, Vector)] =
	textDataRDD1.zipWithIndex.map { case (tokens, id) =>
		 val counts = new HashMap[Int, Double]()
		 tokens.foreach { term =>
			 if (vocab.contains(term)) {
				 val idx = vocab(term)
				 counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
			 }
		 }
		 (id, Vectors.sparse(vocab1.size, counts.toSeq))
	}
	val lda1 = new LDA().setK(numTopics).setMaxIterations(100)
	val ldaModel1 = lda.run(documents1)
	
	val topicIndices1 = ldaModel.describeTopics(maxTermsPerTopic = 20)
	printContent.append("Topics for Worst Airline: "+worstAirline+"\n")
	topicIndices1.foreach { case (terms, termWeights) =>
		 printContent.append("TOPIC:"+"\n")
		 terms.zip(termWeights).foreach { case (term, weight) =>
			printContent.append(s"${vocabArray1(term.toInt)}\t$weight")
			printContent.append("\n")
		 }
		 printContent.append("\n")
	}
	
    val printRdd = spark.sparkContext.parallelize(Seq(printContent))
    printRdd.saveAsTextFile(args(1))


  }
}
