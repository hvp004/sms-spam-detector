package project;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;


public class NBClassifier {
	ArrayList<String> trainingDocs;
	ArrayList<String> allDocs;
	int[] trainingLabels;
	int numClasses;
	int[] classCounts; //number of docs per class
	String[] classStrings; //concatenated string for a given class
	int[] classTokenCounts; //total number of tokens per class
	HashMap<String,Double>[] condProb;
	HashSet<String> vocabulary; //entire vocabuary
	int offset;
	int totalDocs;
	/**
	 * Build a Naive Bayes classifier using a training document set
	 * @param trainDataFolder the training document folder
	 */
	public NBClassifier(String trainDataFolder, int trainPortion)
	{
		trainingDocs = new ArrayList<String>();
		allDocs = new ArrayList<String>();
		numClasses = 2;
		classCounts = new int[numClasses];
		classStrings = new String[numClasses];
		classTokenCounts = new int[numClasses];
		condProb = new HashMap[numClasses];
		vocabulary = new HashSet<String>();
		for(int i=0;i<numClasses;i++){ //just initialization
			classStrings[i] = "";
			condProb[i] = new HashMap<String,Double>();
		}
		totalDocs = 5574;
		offset = Math.round(totalDocs) * trainPortion / 100;
		System.out.println("Numer of Documents For Training: "+offset);
		System.out.println("Numer of Documents For Testing: "+(totalDocs-offset));
		preprocess(trainDataFolder);
		
		for(int i=0;i<numClasses;i++){
			String[] tokens = classStrings[i].split(" ");
			classTokenCounts[i] = tokens.length;
			//collecting the counts
			for(String token:tokens){
				//System.out.println(token);
				vocabulary.add(token);
				if(condProb[i].containsKey(token)){
					double count = condProb[i].get(token);
					condProb[i].put(token, count+1);
				}
				else
					condProb[i].put(token, 1.0);
				//System.out.println(token+" : "+condProb[i].get(token));
			}
		}
		for(int i=0;i<numClasses;i++){
			Iterator<Map.Entry<String, Double>> iterator = condProb[i].entrySet().iterator();
			int vSize = vocabulary.size();
			while(iterator.hasNext())
			{
				Map.Entry<String, Double> entry = iterator.next();
				String token = entry.getKey();
				Double count = entry.getValue();
				//System.out.println(count+1+" / ( "+classTokenCounts[i]+" + "+vSize+" )");
				count = (count+1)/(classTokenCounts[i]+vSize);
				condProb[i].put(token, count);
			}
			//System.out.println("dekho: "+condProb[i]);
		}
		testModel();
	}
	
	private void testModel() {
		int class_ = 0;
		double accuracy = 0.0;
		int count = 0;
		int actual_count = 0;
		int correct_classifications = 0;
		int testdata = 0;
		int[] correct = new int[numClasses];
		int[] test = new int[numClasses];
		String line = "";
		int tabIndex=0;
		String doc = "", type="", msg="";
		for(int i=offset;i<totalDocs;i++){
			line = allDocs.get(i);
			testdata++;
			tabIndex = line.indexOf('\t');
			type = line.substring(0, tabIndex);
			msg = line.substring(tabIndex + 1);
			msg = msg.toLowerCase();
			if(type.equals("ham")){class_ = 0;}
			else{class_ = 1;}
			test[class_]++;
			actual_count = actual_count + class_;
			int label = classify(msg);
			count = count + label;
			//System.out.println(label+"/"+class_+" "+count);
			if(label-class_==0){
				correct_classifications++;
				correct[class_]++;
			}
		}
		System.out.println("Ham Classified: "+correct[0]+"/"+test[0]+" => "+Math.round(correct[0]*1000.0/test[0] )/1000.0);
		System.out.println("Spam Classfied: "+correct[1]+"/"+test[1]+" => "+Math.round(correct[1]*1000.0/test[1] )/1000.0);
		int tp = correct[1];
		int tn = correct[0];
		int fp = test[0] - correct[0];
		int fn = test[1] - correct[1];
		double recall = Math.round(tp * 1000.0 / (tp + fn))/1000.0;
		double precision = Math.round(tp *1000.0 / (tp + fp))/1000.0;
		double f_score = Math.round(2 * precision * recall *100.0/ (precision + recall)) / 100.0;
		System.out.println("Recall: "+recall);
		System.out.println("Precision: "+precision);
		System.out.println("F-Score: "+f_score);
		accuracy = Math.round(correct_classifications * 1000.0 / testdata)/1000.0;
		System.out.println("Accuracy: "+accuracy);
	}

	/**
	 * Classify a test doc
	 * @param doc test doc
	 * @return class label
	 */
	public int classify(String doc){
		int label = 0;
		int vSize = vocabulary.size();
		double[] score = new double[numClasses];
		for(int i=0;i<score.length;i++){
			score[i] = Math.log(classCounts[i]*1.0/trainingDocs.size());
		}
		String[] tokens = doc.split(" ");
		for(int i=0;i<numClasses;i++){
			for(String token: tokens){
				if(condProb[i].containsKey(token)){
					score[i] += Math.log(condProb[i].get(token));
				}
				else{
					score[i] += Math.log(1.0/(classTokenCounts[i]+vSize));
				}
				//System.out.println("token: "+token+" "+score[i]);
			}
		}
		double maxScore = score[0];
		//System.out.println("class 0: "+score[0]);
		for(int i=1;i<score.length;i++){
			//System.out.println("class "+i+": "+score[i]);
			if(score[i]>maxScore){
				label = i;
			}
		}
		return label;
	}
	
	/**
	 * Load the training documents
	 * @param trainDataFolder
	 */
	public void preprocess(String trainDataFolder)
	{
		File folder = new File(trainDataFolder);
		File[] listOfFiles = folder.listFiles();
		for(int i=0;i<listOfFiles.length;i++){
				try{
					BufferedReader reader = new BufferedReader(new FileReader(listOfFiles[i].getAbsoluteFile()));
					String allLines = new String();
					String line = null;
					int tabIndex;
					String type, msg;
					int label;
					int line_count = 0;
					allDocs = new ArrayList<String>();
					while((line=reader.readLine())!=null && !line.isEmpty()){
						allDocs.add(line);
					}
					//System.out.println(allDocs.size());
					//System.out.println(offset);
					for(int s=0;s<offset;s++){
						line = allDocs.get(s);
						tabIndex = line.indexOf('\t');
						type = line.substring(0, tabIndex);
						msg = line.substring(tabIndex + 1);
						msg = msg.toLowerCase();
						if(type.equals("ham")){label = 0;}
						else{label = 1;}
						classStrings[label] += (line + " ");
						classCounts[label]++;
						trainingDocs.add(line);
					}
				}
				catch(IOException ioe){ioe.printStackTrace();}
		}
		/*System.out.println("number of ham tokens "+classStrings[0].length());
		System.out.println("number of spam tokens " +classStrings[1].length());
		*/
		System.out.println("Training Data Class Distribution:");
		System.out.println("Ham: "+classCounts[0]+"("+Math.round(classCounts[0]*10000.0/trainingDocs.size())/100.0+"%)");
		System.out.println("Spam: "+classCounts[1]+"("+Math.round(classCounts[1]*10000.0/trainingDocs.size())/100.0+"%)");
		System.out.println("Total: "+trainingDocs.size());
	}
	
	/**
	 *  Classify a set of testing documents and report the accuracy
	 * @param testDataFolder fold that contains the testing documents
	 * @return classification accuracy
	 */
	public double classifyAll(String testDataFolder)
	{
		File folder = new File(testDataFolder);
		File[] listOfFiles = folder.listFiles();
		double accuracy = 0.0;
		int correct_classifications = 0;
		int testdata = 0;
		for(int i=0;i<listOfFiles.length;i++){
			String doc = "";
			try {
					BufferedReader reader = new BufferedReader(new FileReader(listOfFiles[i].getAbsoluteFile()));
					int count = 0;
					int actual_count = 0;
					int class_=0;
					int[] correct = new int[numClasses];
					int[] test = new int[numClasses];
					String line = "";
					int tabIndex=0;
					allDocs = new ArrayList<String>();
					while((line=reader.readLine())!=null && !line.isEmpty()){
						testdata++;
						tabIndex = line.indexOf('\t');
						String type = line.substring(0, tabIndex);
						String msg = line.substring(tabIndex + 1);
						msg = msg.toLowerCase();
						if(type.equals("ham")){class_ = 0;}
						else{class_ = 1;}
						test[class_]++;
						actual_count = actual_count + class_;
						int label = classify(msg);
						count = count + label;
						//System.out.println(label+"/"+class_+" "+count);
						if(label-class_==0){
							correct_classifications++;
							correct[class_]++;
						}
					}
				int label = classify(doc);
				if(label - class_==0){correct[class_]++;}
			} 
			catch (IOException e){e.printStackTrace();}
		}
		accuracy = Math.round(correct_classifications * 1000.0 / testdata)/1000.0;
		System.out.println("Accuracy: "+accuracy);
		return accuracy;
	}
	
	public static void main(String[] args){		
		String base_folder = "dataset";
		String fname = base_folder;
		System.out.println("Cross Validation Test-Train Split: 70-30 ");
		NBClassifier nbc = new NBClassifier(fname, 70);
		System.out.println();
		System.out.println("Cross Validation Test-Train Split: 80-20 ");
		NBClassifier nbc2 = new NBClassifier(fname, 80);
		System.out.println();
		System.out.println("Cross Validation Test-Train Split: 90-10 ");
		NBClassifier nbc3 = new NBClassifier(fname, 90);
		System.out.println();
	}
}
