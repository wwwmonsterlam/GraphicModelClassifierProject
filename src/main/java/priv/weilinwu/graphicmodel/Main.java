package priv.weilinwu.graphicmodel;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class Main 
{
	public static final Logger logger = LoggerFactory.getLogger(Main.class);
	
	private final static String RESOURCE_PATH = "src/main/resources/";
	private final static String[] DATA_FILE_NAMES = {"classify_d3_k2_saved1.mat",
			"classify_d3_k2_saved2.mat", "classify_d3_k2_saved3.mat",
			"classify_d4_k3_saved1.mat", "classify_d4_k3_saved2.mat",
			"classify_d5_k3_saved1.mat", "classify_d5_k3_saved2.mat",
			"classify_d99_k50_saved1.mat", "classify_d99_k50_saved2.mat",
			"classify_d99_k60_saved1.mat", "classify_d99_k60_saved2.mat"};
	
    public static void main( String[] args ) throws IOException
    {
    	for(int i = 0; i < 11; i++) {
    		Matrix[] dataSet = Utils.dataLoader(RESOURCE_PATH + DATA_FILE_NAMES[i]);
    		Utils.scaleData(dataSet);
	    	Matrix[] trainingSet = Utils.getTrainingSet(dataSet);
//	    	logger.debug(trainingSet[0].toString());
	    	Matrix[] testingSet = Utils.getTestingSet(dataSet);
	    	
	    	LogisticClassifier logisticClassifier = new LogisticClassifier(trainingSet, testingSet);
	    	
	    	logger.debug("result of the {" + (i + 1) + "}th dataset:");
	    	logger.debug("correction rate: " + logisticClassifier.getCorrectionRateUsingTestingSetUsingOriginAsStartingPoint());
	    	if(i <= 6) {
	    		logger.debug("final theta is: " + logisticClassifier.getTheta().toString());
	    	}
    	}
//    	for(int i = 0; i < 10; i++) {
//    		double a = (Math.random() - 0.5) * Math.pow(10, 300);
//    		logger.debug(a + "");
//    	}
    }
}
