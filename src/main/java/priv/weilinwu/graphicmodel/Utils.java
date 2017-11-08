package priv.weilinwu.graphicmodel;

import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.jmatio.ImportMatrixMAT;

public class Utils {
	public static final Logger logger = LoggerFactory.getLogger(Utils.class);
	
	public static Matrix[] dataLoader(String dataPath) throws IOException {
		File file = new File(dataPath);
		Matrix[] matrixs = new Matrix[2];
		matrixs[0] = ImportMatrixMAT.fromFile(file, "class_1");
		matrixs[1] = ImportMatrixMAT.fromFile(file, "class_2");
		
//		logger.debug("The size of the matrix is " + matrixs[0].getRowCount() + " by " 
//				+ matrixs[0].getColumnCount());
//		logger.debug(matrixs[0].toString());
//		logger.debug(matrixs[1].toString());
		
		return matrixs;
	}
	
	public static Matrix[] getTrainingSet(Matrix[] matrixs) {
		long rowCount = matrixs[0].getRowCount();
		long columnCount = matrixs[0].getColumnCount();	
		Matrix[] trainingSet = new Matrix[2];
		
		trainingSet[0] = matrixs[0].subMatrix(Ret.NEW, 0, 0, rowCount - 1, columnCount / 5 * 4 - 1);
		trainingSet[1] = matrixs[1].subMatrix(Ret.NEW, 0, 0, rowCount - 1, columnCount / 5 * 4 - 1);
		
//		logger.debug("The size of training set is: " + trainingSet[0].getRowCount() + 
//				" by " + trainingSet[0].getColumnCount());
		
		return trainingSet;
	}
	
	public static Matrix[] getTestingSet(Matrix[] matrixs) {
		long rowCount = matrixs[0].getRowCount();
		long columnCount = matrixs[0].getColumnCount();	
		Matrix[] testingSet = new Matrix[2];
		
		testingSet[0] = matrixs[0].subMatrix(Ret.NEW, 0, columnCount / 5 * 4, rowCount - 1, columnCount - 1);
		testingSet[1] = matrixs[1].subMatrix(Ret.NEW, 0, columnCount / 5 * 4, rowCount - 1, columnCount - 1);
		
//		logger.debug("The size of testing set is: " + testingSet[0].getRowCount() + 
//				" by " + testingSet[0].getColumnCount());
		
		return testingSet;
		
	}
	
	// scale each data so that each one is between -0.5 and 0.5
	public static void scaleData(Matrix[] m) {
		long rowCount = m[0].getRowCount();
		long columnCount = m[0].getColumnCount();
		
		for(int z = 0; z <= 1; z++) {
			for(int r = 0; r < rowCount; r++) {
				double sum = m[z].getAsDouble(r, 0);
				double max = sum;
				double min = sum;
				
				for(int c = 1; c < columnCount; c++) {
					double x = m[z].getAsDouble(r, c);
					sum += x;
					max = Math.max(x, max);
					min = Math.min(x, min);
				}
				
				double mean = sum / columnCount;
				double scope = max - min;
				
				for(int c = 0; c < columnCount; c++) {
					double x = m[z].getAsDouble(r, c);
					m[z].setAsDouble((x) / scope, r, c);
				}
			}
		}
		
	}
}
