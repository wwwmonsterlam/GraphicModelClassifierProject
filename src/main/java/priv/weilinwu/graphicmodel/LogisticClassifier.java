package priv.weilinwu.graphicmodel;

import java.io.ObjectInputStream.GetField;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;
import org.ujmp.core.doublematrix.calculation.entrywise.creators.Zeros;

public class LogisticClassifier {
	public static final Logger logger = LoggerFactory.getLogger(LogisticClassifier.class);
	
	private Matrix[] trainingSet;
	private Matrix[] testingSet;
	private Matrix theta;
	private Matrix gradientDescendDirection;
	private long featureCount;
	private long trainingSampleCount;
	private final double tolerance = 0.1;
	
	LogisticClassifier(Matrix[] trainingSet, Matrix[] testingSet) {
		// For calculation simplicity, add one to each column
		// In this way, theta can direct multiply each sample vector in the data set
		this.trainingSet = addOneToEachColumn(trainingSet);
		this.testingSet = addOneToEachColumn(testingSet);
//		logger.debug(this.testingSet[0].toString());
		this.featureCount = trainingSet[0].getRowCount();
		this.trainingSampleCount = trainingSet[0].getColumnCount();
		theta = DenseMatrix.Factory.zeros(this.featureCount + 1 , 1);
//		logger.debug(theta.toString());
		
		generateNewGradientDescendDirection();
	}
	
	public Matrix getTheta() {
		return theta;
	}
	
	private Matrix[] addOneToEachColumn(Matrix[] m) {
		Matrix temp = DenseMatrix.Factory.ones(1 , m[0].getColumnCount());
		Matrix modifiedClass0 = temp.appendVertically(Ret.NEW, m[0]);
		Matrix modifiedClass1 = temp.appendVertically(Ret.NEW, m[1]);
		Matrix[] result = {modifiedClass0, modifiedClass1};
		
		return result;
	}
	
	// derivative of the objective function
	public void generateNewGradientDescendDirection() {
		Matrix sum = DenseMatrix.Factory.zeros(featureCount + 1, 1);
		
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < trainingSampleCount; i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
				sum = sum.plus(xi.times(z - getValueOfLogisticFunction(theta, xi)));
//				logger.debug(xi.times(z - getValueOfLogisticFunction(theta, xi)).toString());
			}
		}
		
//		logger.debug(sum.toString());
		
		this.gradientDescendDirection = sum;
	}
	
	public double getValueOfLogisticFunction(Matrix theta, Matrix x) {
		double productOfThetaAndX = theta.transpose().mtimes(x).getAsDouble(0, 0);
		double value = 1.0 / (1 + Math.exp(-1 * productOfThetaAndX));
		
//		logger.debug("product Of Theta And X is " + productOfThetaAndX);
//		logger.debug("the value is " + value);
		
		return value;
	}
	
	// get (local) optimal step size
	public double getOptimalStepSize() {
		double[] pointPair = getInitialPointPair();
		updatePointPairUntillDistanceWithinTolerance(pointPair);

		double stepSize = (pointPair[0] + pointPair[1]) / 2.0;
		return stepSize;
	}
	
	public double[] getInitialPointPair() {
		double startingPoint = 0;
		double searchSize = 0.01;
		double left = 0;
		double right = 0;
		
		if(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(startingPoint) <= 0) {
			right = startingPoint;
			left = right - searchSize;
			while(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(left) <= 0) {
//				logger.debug("searching for left point of the step size function, left = " + left);
//				logger.debug("the derivative value is: " + getValueOfObjectiveFunctionderivativeWithRespectToStepSize(left));
				searchSize *= 2;
				left = right - searchSize;
			}
		} else {
			left = startingPoint;
			right = left + searchSize;
			while(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(right) > 0) {
//				logger.debug("searching for right point of the step size function, right = " + right);
//				logger.debug("the derivative value is: " + getValueOfObjectiveFunctionderivativeWithRespectToStepSize(right));
				searchSize *= 2;
				right = left + searchSize;
			}
		}
		
		double[] initialPointPaire = {left, right};
		return initialPointPaire;
	}
	
	public void updatePointPairUntillDistanceWithinTolerance(double[] pointPair) {
		double left = pointPair[0];
		double right = pointPair[1];
		if(right - left < tolerance * 0.1) {
			return;
		}
		
		double middle = (right - left) / 2.0 + left;
		if(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(middle) <= 0) {
			right = middle;
		} else {
			left = middle;
		}
		
		pointPair[0] = left;
		pointPair[1] = right;
		updatePointPairUntillDistanceWithinTolerance(pointPair);
	}
	
	// return true if theta is updated, return false if the difference is within tolerance
	public boolean updateTheta() {
		generateNewGradientDescendDirection();
		if(isWithinTolerance(gradientDescendDirection)) {
			return false;
		} else {
			double stepSize = getOptimalStepSize();
			theta = theta.plus(gradientDescendDirection.times(stepSize));
			return true;
		}
	}
	
	// return true if every entry is no larger than the tolerance value
	public boolean isWithinTolerance(Matrix m) {
//		logger.debug("the difference is: " + m.toString());
		for(int i = 0; i <= featureCount; i++) {
			if(Math.abs(m.getAsDouble(i, 0)) > tolerance) {
				return false;
			}
		}
		return true;
	}
	
	// this is used to help find optimal step size
	public double getValueOfObjectiveFunctionderivativeWithRespectToStepSize(double stepSize) {
		double sum = 0;
		
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < trainingSampleCount; i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
//				logger.debug("xi is \n" + xi.toString());
				Matrix newTheta = theta.plus(gradientDescendDirection.times(stepSize));
//				logger.debug("new theta is" + newTheta.toString());
				sum += gradientDescendDirection.transpose().mtimes(xi).getAsDouble(0, 0) *
						(z - getValueOfLogisticFunction(newTheta, xi));
//				logger.debug("test point 1: " + gradientDescendDirection.transpose().mtimes(xi).toString());
//				logger.debug("gradient descend direction: " + gradientDescendDirection.toString());
//				logger.debug("Now the sum is: " + sum);
			}
		}
		
		return sum;
	}
	
	public double getValueOfObjectiveFunction() {
		double sum = 0;
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < trainingSampleCount; i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
				sum += z * Math.log(getValueOfLogisticFunction(theta, xi)) + 
						(1 - z) * Math.log(1 - getValueOfLogisticFunction(theta, xi));
			}
		}
		
		return sum;
	}
	
	public void train() {
		int i = 1;
		while(updateTheta()) {
//			logger.debug("The {" + i + "} update finished!");
//			logger.debug("the value of objective function is: " + getValueOfObjectiveFunction());
			if(i % 10 == 0) {
				System.out.print(".");
			}
			i++;
		}
		System.out.print("\n");
		logger.debug("Total iteration count: " + i);
	}
	
	public double getCorrectionRateUsingTestingSetUsingOriginAsStartingPoint() {
		train();
		long sizeOfTesingSet = testingSet[0].getColumnCount() * 2;
		int errorPredictionCount = 0;
		
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < testingSet[0].getColumnCount(); i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
				int prediction = theta.transpose().mtimes(xi).getAsDouble(0, 0) <= 0 ? 0 : 1;
				if(prediction != z) {
					errorPredictionCount++;
				}
			}
		}
		
		return (double)(sizeOfTesingSet - errorPredictionCount) / sizeOfTesingSet;
	}
	
	public double getCorrectionRateUsingTestingSetUsingMultipleRandomStartingPoint(int randomStartingPointAmount) {
		if(randomStartingPointAmount <= 0) {
			randomStartingPointAmount = 1;
		}
		
		long sizeOfTesingSet = testingSet[0].getColumnCount() * 2;
		double correctionRate = 0;
		
		while(randomStartingPointAmount-- > 0) {
			train();
			int errorPredictionCount = 0;
			
			for(int z = 0; z <= 1; z++) {
				for(int i = 0; i < testingSet[0].getColumnCount(); i++) {
					// Get the i th sample
					Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
					int prediction = theta.transpose().mtimes(xi).getAsDouble(0, 0) <= 0 ? 0 : 1;
					if(prediction != z) {
						errorPredictionCount++;
					}
				}
			}
			
			double tempRate = (double)(sizeOfTesingSet - errorPredictionCount) / sizeOfTesingSet;
			correctionRate = Math.max(correctionRate, tempRate);

			setThetaUsingRandomNumber();
		}
		
		return correctionRate;
	}
	
	public void setThetaUsingRandomNumber() {
		for(int i = 0; i < 4; i++) {
			theta.setAsDouble((Math.random() - 0.5) * Math.pow(10, 2), i, 0);
		}
	}
}

