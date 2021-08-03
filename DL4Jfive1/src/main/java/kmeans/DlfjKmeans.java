package kmeans;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class DlfjKmeans  {


    public static void main(String[] args) throws IOException {
        CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\predictionsfile1.csv"), ',');

        List<String[]> s;
         s=csvReader.readAll();
         Iterator<String[]> it=s.iterator();
         //it.next();

        //recordReaderTrain.
        List<Point>points=new ArrayList<>();

        List<INDArray>vectors = new ArrayList<>();
        it.next();
        while (it.hasNext()){
              String[] row =  it.next();
             // Double[] d = (Double[]) row;
             /*double[] doubleValues = Arrays.stream(row)
                     .mapToDouble(Double::parseDouble)
                     .toArray();*/
             Double[] doubleValues = Arrays.stream(row)
                     .map(Double::valueOf)
                     .toArray(Double[]::new);
             INDArray r = Nd4j.createFromArray(doubleValues);
             vectors.add(r);
             for(int i=0;i< row.length;i++)
                System.out.print(row[i]+" ");
             System.out.println("\n");
        }
        System.out.println("-------------");
        points = Point.toPoints(vectors);

        System.out.println(points.size());

        int maxIterationCount = 5;
        int clusterCount = 3;
        boolean useKplusplus = true;
        //String distanceFunction = "cosinesimilarity";
        KMeansClustering kmc = KMeansClustering.setup(clusterCount,maxIterationCount, Distance.EUCLIDEAN,useKplusplus);
        ClusterSet cs = kmc.applyTo(points);
        for(int i=0;i<points.size();i+=10){
            Pair<Cluster, Double> c = cs.nearestCluster(points.get(i));
           // c.getFirst()
            System.out.println(cs.nearestCluster(points.get(i)).toString());
        }


        List<Cluster> clsterLst = cs.getClusters();

        System.out.println("\nCluster Centers:");
        for(Cluster c: clsterLst) {

            Point center = c.getCenter();
            System.out.println(center);
            System.out.println("----");
            System.out.println(center.getArray());
           /* System.out.println("--label--");
            System.out.println(center.getLabel());
            System.out.println("---id-");
            System.out.println(center.getId());*/
        }


        //recordReaderTrain.initialize(new  FileSplit(File("src/main/resources/data/Data.csv")));
       // DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 3, 2);

    }




}
