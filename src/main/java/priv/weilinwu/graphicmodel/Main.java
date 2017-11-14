package priv.weilinwu.graphicmodel;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
    	char choice = getUserChoice();
    	int iterationCount = choice == 'a'? 1 : 100; 
    	double[] accuracies = new double[DATA_FILE_NAMES.length];
    	for(int i = 0; i < DATA_FILE_NAMES.length; i++) {
    		
    		String filePath = RESOURCE_PATH + DATA_FILE_NAMES[i];
    		Matrix[] dataSet = Utils.dataLoader(filePath);
    		Utils.scaleData(dataSet);
	    	Matrix[] trainingSet = Utils.getTrainingSet(dataSet);
	    	Matrix[] testingSet = Utils.getTestingSet(dataSet);
	    	
	    	LogisticClassifier logisticClassifier = new LogisticClassifier(trainingSet, testingSet);
	    	
	    	logger.debug("Running logistic classifier using dataset {" + filePath + "} :");
	    	accuracies[i] = logisticClassifier.getCorrectionRateUsingTestingSetUsingMultipleRandomStartingPoint(iterationCount);
	    	logger.debug("######################################################################################");
	    	logger.debug("correction rate (" + DATA_FILE_NAMES[i] + "): " + accuracies[i] + "\n");
    	}
    	
    	summary(accuracies);
    }
    
    public static void summary(double[] accuracies) {
    	logger.debug("");
    	logger.debug("####################################################");
    	logger.debug("Summary of classification using Logistic classifier");
    	logger.debug("####################################################");
    	for(int i = 0; i < DATA_FILE_NAMES.length; i++) {
	    	logger.debug("correction rate (" + DATA_FILE_NAMES[i] + "): " + accuracies[i]);
    	}
    }
    
    public static char getUserChoice() throws IOException {
    	logger.debug("##########################################################################################");
    	logger.debug("Please Select how times of logistic classification would you like to run for each dataset");
    	logger.debug("##########################################################################################");
    	logger.debug("");
    	logger.debug("If you input 'a', it would run for once (recommended for time consuming purpose)");
    	logger.debug("If you input 'b', it would run for 100 times with different initialization (random), consuming lots of time but with better accuracy");
    	return (char) System.in.read();
    }
}
