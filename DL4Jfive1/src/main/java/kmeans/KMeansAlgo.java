/*
package kmeans;

import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;


public class KMeansAlgo {

    private static final Logger log = LoggerFactory.getLogger(KMeansAlgo.class);

    public static void main(String[] args) throws Exception {
        String datafilepath = "/raw_sentences.txt"; //This has ~99000 single-sentence paragraphs/docs
        ClassPathResource resource = new ClassPathResource(datafilepath);
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);
        InMemoryLookupCache cache = new InMemoryLookupCache();
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        StopWatch sw = new StopWatch();
        */
/*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.
              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
        *//*

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)  */
/*length of a paragraph vector*//*

                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();


        //1. create a kmeanscluster instance
        int maxIterationCount = 5;
        int clusterCount = 10;
        String distanceFunction = "cosinesimilarity";
        KMeansClustering kmc = KMeansClustering.setup(clusterCount, maxIterationCount, distanceFunction);

        //2. iterate over rows in the paragraphvector and create a List of paragraph vectors
        List<INDArray> vectors = new ArrayList<INDArray>();
        for (String word : vec.vocab().words()) {
            vectors.add(vec.getWordVectorMatrix(word));
        }
        log.info(vectors.size() + " vectors extracted to create Point list");
        List<Point> pointsLst = Point.toPoints(vectors);
        log.info(pointsLst.size() + " Points created out of " + vectors.size() + " vectors");

        log.info("Start Clustering " + pointsLst.size() + " points/docs");
        sw.reset();
        sw.start();
        ClusterSet cs = kmc.applyTo(pointsLst);
        sw.stop();
        System.out.println("Time taken to run clustering on " + vectors.size() + " paragraphVectors: " + sw.getTime());
        vectors = null;
        pointsLst = null;

        log.info("Finish  Clustering");

        List<Cluster> clsterLst = cs.getClusters();

        System.out.println("\nCluster Centers:");
        for(Cluster c: clsterLst) {
            Point center = c.getCenter();
            System.out.println(center.getId());
        }

        log.info("Trying to classify a point that was used for generating the Clusters");
        double[] nesVec = vec.getWordVector("DOC_400");
        Point newpoint = new Point("myid", "mylabel", nesVec);
        PointClassification pc = cs.classifyPoint(newpoint);
        System.out.println(pc.getCluster().getCenter().getId());

        System.out.println("\nEnd Test");

    }
}
*/
