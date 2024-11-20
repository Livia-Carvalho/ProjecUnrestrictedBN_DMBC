import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.BIFReader;
import weka.classifiers.bayes.net.EditableBayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Copyable;
import weka.core.FastVector;
import weka.core.Instances;
import classifiers.bayes.net.search.local.DMBC;

/*** Classe MainUnrestrictedBN ************************************************
 *   
 *   Classe responsavel pela uniao de redes bayesianas induzidas pelo algoritmo
 *   DMBC para gerar uma rede bayesiana irrestrita.
 *   
 ****/
public class MainUnrestrictedBN {
	
	// guarda as instancias do conjunto de dados, lidas a partir do arquivo.
	private Instances instances;
	// variavel que representa as redes bayesianas induzidas pelo DMBC.
	private EditableBayesNet net;
	// variavel que executa o algoritmo DMBC.
	private DMBC dmbcAlgorithm;
	int parentNumber;
	int minWeight;
	// armazena a ordenacao de variaveis passada como parametro pelo usuario.
	int vetVariable[];
	private Evaluation eval;
	int indexclassVariable;
	private int gFunctionCounter = 0;
	
	static public void main(String args[])
	{
		new MainUnrestrictedBN(args);					
	}
	
	public MainUnrestrictedBN(String[] args)
	{	
		try
		{
			if(args.length != 3)
			{
				System.out.println("Erro de sintaxe!");
				System.out.println("Sintaxe correta: Unrestricted_DMBC.jar " +
						           "'nome do arquivo da base de dados'.arff " +						           
						           "'nome do arquivo de ordenacao de variaveis' " +						      
						           "'numero maximo de pais'" );//+						      
//						           "'peso minimo de aresta'");
				return;
			}
			System.out.println("Executando o algoritmo de inducao de RB irrestrita "
					            + "a partir do DMBC...");	
			
			// variaveis de inicializacao....			
			// guarda o tempo de inicio da execucao....
			long initialTime = System.currentTimeMillis();
			// guarda o nome do arquivo de entrada.
			String inputFileName = args[0];
			// armazena as instancias de treinamento.
			instances = readDataSetFile(inputFileName);
			// armazena a ordenacao de variaveis fornecida pelo usuario.
			
			vetVariable = readOrderingFile(args[1]);
//			vetVariable = generateOrderingVector();
			
			indexclassVariable = vetVariable[0];
			parentNumber = Integer.parseInt(args[2]);
			/*
			minWeight = Integer.parseInt(args[2]);
			if(minWeight < 1) {
				System.out.println("Erro! O valor de peso minimo das arestas deve ser maior que 0");
				return;
			}
			*/
			minWeight = 1;
			
			// guarda a rede bayesiana irrestrita que sera construida a partir 
			// das redes induzidas pelo DMBC.
			EditableBayesNet unrestrictedNet = new EditableBayesNet(instances);
			net = new EditableBayesNet(instances);
			// inicia o contador de chamadas da funcao g.
			//dmbcAlgorithm.setGFunctionCounter(0);
			// constroi uma rede bayesiana usando o DMBC para cada variavel  
			// da base identificada como sendo a classe. Dessa forma, a rede
			// aprendida representa o markov blanket de cada variavel classe.

			int numAttributes = instances.numAttributes();
			int[][] edgesCount = new int[numAttributes][numAttributes];
			for (int i = 0; i < numAttributes; i++) {
			    for (int j = 0; j < numAttributes; j++) {
			        edgesCount[i][j] = 0;
			    }
			}
			
			for(int i=0; i<numAttributes; i++)
			{
				// vetor de ordenacao das variaveis na iteracao atual
				int vetOrderingLocal[];
				// variavel que guarda o indice da varivel classe no vetor de ordenacao recebido do usuario.
				int classIndexInVet = i;
//				System.out.println("Aplicando o DMBC na classe: " + classIndexInVet);
				// variavel que guarda o valor numerico da variavel classe, presente no vetor de ordenacao.
				int classValueNum = vetVariable[i];
				// configura o indice da variavel classe (vai de 0 ate o numero de atributos da base de dados - 1).
				instances.setClassIndex(classValueNum);
				// armazena a ordenacao de variaveis a ser seguida, sempre tendo a classe na primeira posicao do vetor.
				vetOrderingLocal = changeOrdering(vetVariable, classValueNum);
				// dmbc utiliza a ordenacao passada como parametro de entrada, tendo a variavel classe na primeira posicao.
				// aprende a estrutura da rede atraves do DMBC.
				buildBayesianNetwork(instances, vetOrderingLocal);
				
				//pra cada atributo
				for (int a = 0; a < numAttributes; a++) {
					//pra cada filho do atributo
					for (int c : net.getChildren(a)) {
						//adiciona uma aresta na matriz
						edgesCount[a][c]++;
					}
				}
				
				
//				net.getChildren(i);
//				System.out.println(net.getChildren(i));
				/*
				for (int k = 0; k < numAttributes; k++) {
				    for (int j = 0; j < numAttributes; j++) {
				    	System.out.print(edgesCount[k][j]);
				    }
				    System.out.println();
				}
				*/

				// salva as redes em arquivo
				
//				saveFileNet(net, i, inputFileName/*, gValue*/);
				/* constroi a rede bayesiana irrestrita a partir das redes do DMBC para cada atributo
				 * inserido como classe na primeira posicao do vetor de ordenacao */
//				buildUnrestrictedNet(unrestrictedNet, net, vetVariable, classValueNum, classIndexInVet);								
			}
			
			for (int i = 0; i < numAttributes; i++) {
				for (int j = i+1; j < numAttributes; j++) { // percorrendo apenas metade da matriz
			    	if(edgesCount[i][j] != 0) {
			    		if(edgesCount[i][j] == edgesCount[j][i]) {
				    		edgesCount[i][j] = 0;
				    		edgesCount[j][i] = 0;
			    		} else if(edgesCount[i][j] > edgesCount[j][i]){
			    			edgesCount[i][j] -= edgesCount[j][i];
				    		edgesCount[j][i] = 0;			    			
			    		} else {
			    			edgesCount[j][i] -= edgesCount[i][j];
				    		edgesCount[i][j] = 0;
			    		}
			    	}
			    }
			}
			
			//APLICANDO PESOS NA MATRIZ
			for (int i = 0; i < numAttributes; i++) {
//			    System.out.println();
			    for (int j = 0; j < numAttributes; j++) {
//			    	System.out.print(edgesCount[i][j]);
			    	
			    	if (i != j && edgesCount[i][j] > (minWeight - 1)) {
			    		edgesCount[i][j] = 1;
			        }else {
			        	edgesCount[i][j] = 0;
			        }
			    	
			    }
			}
			
			saveMatrixToFile(edgesCount, inputFileName);
			
			/*
			for (int i = 0; i < numAttributes; i++) {
//			    System.out.println();
			    for (int j = 0; j < numAttributes; j++) {
//			    	System.out.print(edgesCount[i][j]);
			    	
			    	if (i != j && edgesCount[i][j] == 1) {
			            unrestrictedNet.addArc(i, j);
			        }
			    }
			}
			*/

			
						
			// ***********************************************
			
			//quantidade total de chamadas a funcao G
			System.out.println("\nChamadas a funcao G: " + gFunctionCounter);
			// guarda o tempo de termino da execucao
			long finalTime = System.currentTimeMillis();
			// Calcula tempo de execucao do algoritmo DMBC.
			double runTime = (double)((finalTime - initialTime));
			System.out.println("\nTempo de execucao (ms): " + runTime);
			writeToCSV(inputFileName, runTime, gFunctionCounter);
			// salva a rede irrestrita em arquivo
			System.out.println("Rede Bayesiana irrestrita de peso minimo " + minWeight + " concluida!!!");
			saveNet(unrestrictedNet, inputFileName, runTime);
			
			//System.out.println("\n***** Realizando a classificacao... *****");
			// realiza a classificacao
			//trainAndTestSplitClassification(unrestrictedNet);
			
			// Salva os valores de classificacao
			/*
			FileWriter fileClassification = new FileWriter("Classification_UnrestrictedBN_DMBC.txt", true);
			PrintWriter fclass = new PrintWriter(fileClassification, true);
			fclass.println(eval.toSummaryString("\nResults\n======\n", false));
			fclass.println(eval.toClassDetailsString());
			fclass.println(eval.toMatrixString(("\n=== Confusion matrix ===\n")));
			fclass.close();
			*/
			
			//System.out.println("Numero de chamadas a funcaoo g: " + dmbcAlgorithm.getGFunctionCounter());
			System.out.println("\nExecucao do DMBC irrestrito concluida com sucesso!");
//			showNet(buildUnrestrictedNet.graph(), gValue);

		}
		catch (Exception e) 
		{
			e.printStackTrace();
		}		
	}
	
	private Instances readDataSetFile(String dataSetFileName) 
	{
		Instances instances = null;
		try 
		{
			System.out.println("Lendo o arquivo da base de dados: " + dataSetFileName);
			instances = getInstances(dataSetFileName);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			System.out.println("A leitura do arquivo " + dataSetFileName + 
	                           " nao foi possivel!");
		}
		return instances;
	}
	
	/**
	 * @param arffName
	 *            - O caminho completo do arquivo Arff
	 * @return - Um objeto Instances (do Weka) que � uma cole��o de v�rios
	 *         objetos Instance (cada Instance representa uma linha do arquivo
	 *         ARFF)
	 * @throws Exception
	 *             - se algo der errado, lan�a uma Exce�ao gen�rica
	 */
	private static Instances getInstances(String arffName) throws Exception {
		// FileReader e uma classe do Java para leitura de arquivos. No
		// construtor dela, passamos o caminho do arquivo que queremos ler.
		FileReader arffReader = new FileReader(arffName);

		// Instances e uma classe do Weka, e no construtor dela podemos passar
		// um objeto FileReader (que construimos na linha acima)
		// Dai o Weka recupera os dados do arquivo pelo FileReader e
		// monta uma colecao de Instance num objeto Instances
		Instances instances = new Instances(arffReader);
		arffReader.close(); // fechamos o leitor de arquivo

		return instances; // retorna o objeto de Instances
	}
	
	private int[] readOrderingFile(String orderingFileName) {
	    // Cria um array de inteiros chamado 'vet' com o tamanho baseado na quantidade de atributos
	    int vet[] = new int[instances.numAttributes()];

	    // Cria uma instância da classe File que representa o arquivo a ser lido
	    File orderingFile = new File(orderingFileName);

	    try {
	        // Cria um objeto BufferedReader para leitura do arquivo
	        BufferedReader reading = new BufferedReader(new FileReader(orderingFile));
	        String line;
	        int variable;
	        int position = 0;

	        // Entra em um loop para ler cada linha do arquivo
	        while ((line = reading.readLine()) != null) {
	            // Divide a linha em tokens (números inteiros)
	            StringTokenizer st = new StringTokenizer(line);

	            // Entra em um loop interno para processar cada token
	            while (st.hasMoreTokens()) {
	                // Converte o token em um número inteiro e armazena em 'variable'
	                variable = (Integer.parseInt(st.nextToken()));

	                // Armazena 'variable' na posição atual do array 'vet'
	                vet[position] = variable;

	                // Incrementa a posição no array 'vet'
	                position++;
	            }
	        }

	        // Fecha o BufferedReader após a leitura do arquivo
	        reading.close();
	    } catch (Exception e) {
	        // Trata exceções, imprime o erro no console e exibe uma mensagem de erro
	        e.printStackTrace();
	        System.out.println("A leitura do arquivo " + orderingFileName +
	                           " nao foi possivel!");
	    }

	    // Retorna o array 'vet' preenchido com os números lidos do arquivo
	    return vet;
	}

	
	private int[] generateOrderingVector() {
		
	    int numAttributes = instances.numAttributes();
	    int[] vet = new int[numAttributes];
	    
	    for (int i = 0; i < numAttributes; i++) {
	        vet[i] = i;
	    }
	    
	    return vet;
	}
	
	public int[][] loadMatrixFromFile(String fileName) {
	    int[][] matrix = null;
	    try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
	        String line;
	        int rowNum = 0;
	        while ((line = reader.readLine()) != null) {
	            String[] values = line.split(",");
	            if (matrix == null) {
	                // Inicializa a matriz na primeira linha lida
	                matrix = new int[values.length][values.length];
	            }
	            for (int colNum = 0; colNum < values.length; colNum++) {
	                matrix[rowNum][colNum] = Integer.parseInt(values[colNum]);
	            }
	            rowNum++;
	        }
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	    return matrix;
	}
	
	public void saveMatrixToFile(int[][] matrix, String inputFileName) {
		
		String outputName = "Unrestricted-DMBC-Matrix_" + minWeight + "_";				
		String outputFileName = inputFileName.replace(".arff", ".txt");
		outputFileName = outputName.concat(outputFileName);
		
	    try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName))) {
	        for (int i = 0; i < matrix.length; i++) {
	            for (int j = 0; j < matrix[i].length; j++) {
	                writer.write(String.valueOf(matrix[i][j]));
	                if (j < matrix[i].length - 1) {
	                    writer.write(","); // Use v�rgula (ou outro delimitador) para separar os elementos
	                }
	            }
	            writer.newLine(); // Nova linha ap�s cada linha da matriz
	        }
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}

	
	public void buildBayesianNetwork(Instances instances, int [] _vetVariable) throws Exception
	{
		dmbcAlgorithm = new DMBC();
		
		
		// armazena as informacoes da base de dados.
		net.m_Instances = instances;
		// parametros de inicializacao do DMBC:
		// 1 - inicia como naive bayes ou nao:
		dmbcAlgorithm.setInitAsNaiveBayes(false);
		// 2 - utiliza o markov blanket ou nao:
		dmbcAlgorithm.setMarkovBlanketClassifier(false);
		// 3 - configura o numero maximo de pais:
		dmbcAlgorithm.setMaxNrOfParents(parentNumber);
		// 4 - utiliza a ordenacao inicial ou nao:
		dmbcAlgorithm.setRandomOrder(false);	
		// 5 - utiliza a ordenacao passada pelo usuario.
		dmbcAlgorithm.setVariableOrdering(_vetVariable);
		//6 - inicia o contador de chamadas da funcao g.
		dmbcAlgorithm.setGFunctionCounter(0);
		// constroi a estrutura da rede e calcula as probabilidades.
		net.setSearchAlgorithm(dmbcAlgorithm);
		// passa ao DMBC a ordenacao de variaveis que sera usada no aprendizado.
		//dmbcAlgorithm.setVariableOrdering(vetOrdering);
		// aprende a estrutura da rede.
		net.initStructure();
		net.initCPTs();
		net.buildStructure();
		// calcula os parametros numericos
		net.estimateCPTs();
		//incrementa o contador total de chamadas a funcao g
		gFunctionCounter += dmbcAlgorithm.getGFunctionCounter();
//		System.out.println("Chamadas a funcao G no dmbc: " + dmbcAlgorithm.getGFunctionCounter());
		// retorna o valor de g.
		//gValue = net.measureBayesScore();
	}
	
	// Funcao que altera a variavel classe da ordenacao de variaveis fornecida  
	// pelo usuario.
	private int[] changeOrdering(int _vet[], int _classIndex)
	{				
		int vet[] = new int[instances.numAttributes()];
		int variable;
		int position=0;
		
		initVetPosition(vet);
		vet[position] = _classIndex;
		position++;
		for(int i=0; i<_vet.length; i++)	
		{
			variable = _vet[i];
			if( !(isPresent(vet, variable)) )
			{
				vet[position] = variable;
				position++;
			}
	
		}
		return vet;
	}
	
    public void writeToCSV(String datasetName, double runTime, int gFunctionCounter) {
        try (FileWriter writer = new FileWriter(datasetName.substring(0, datasetName.length() - 5) + "_DMBBN_time_Gfunc.csv", true)) {
        	
        	File file = new File(datasetName.substring(0, datasetName.length() - 5) + "_DMBBN_time_Gfunc.csv");
        	if (file.length() == 0) {
                // Escreve a linha de cabe�alho se o arquivo estiver vazio
                writer.append("runtime, Gfunc\n");
            }
        	
            writer.append(String.valueOf(runTime)).append(",");
            writer.append(String.valueOf(gFunctionCounter)).append("\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
	
	/*
	
	private BayesNet readFileNet(int i, String inputFileName) throws Exception
	{
		// Nome do arquivo a ser lido.
		String inputName = "DMBC_";				
		inputName = inputName.concat(String.valueOf(i));
		inputName = inputName.concat("_");
		inputFileName = inputFileName.replace(".arff", ".xml");
		inputFileName = inputName.concat(inputFileName);
		// Le do arquivo a rede bayesiana aprendida pelo DMBC.
		BIFReader reader = new BIFReader();
		BayesNet dmbcNet = reader.processFile(inputFileName);
		
		return dmbcNet;
	}
	private void arcsMatrix(int[][] edgesCount, int numAttributes)
	{		
		try {
			//pra cada atributo
			for (int iAttribute = 0; iAttribute < numAttributes; iAttribute++) {
				//pra cada pai do atributo
				for (int iParent = 0; iParent < m_ParentSets[iAttribute].getNrOfParents(); iParent++) {
					//index do pai na rede
					int nParent = m_ParentSets[iAttribute].getParent(iParent);
					edgesCount[nParent][iAttribute]++;
				}
			}
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
	  }
	  */
	
	private void buildUnrestrictedNet(EditableBayesNet _unrestrictedNet, 
			                          EditableBayesNet _net,
			                          int[] _vetVariable,
			                          int _classValueNum,
			                          int _classIndexInVet) throws Exception 
	{
		// guarda o nome do node classe da estrutura gerada pelo DMBC.
		String nodeName = _net.getNodeName(_classValueNum);
		//System.out.println("construindo a rede irrestrita: " + nodeName);
		// percorre a rede induzida pelo DMBC em busca do node classe.
		for(int i=0; i<_net.getNrOfNodes(); i++)
		{
			// verifica se o node analisado eh o node classe
			if(_net.getNodeName(i)==nodeName)
			{
				//System.out.println("Node classe analisado: " + nodeName);
				// guarda os nodes filhos da classe.
				ArrayList<Integer> children = _net.getChildren(i);
				verifyChildren(children, _vetVariable, _classIndexInVet);
				// insere os nodes filhos da classe no node correspondente da
				// na nova rede irrestrita sendo construida.
				_unrestrictedNet.addArc(nodeName, children);
				children.clear();
				i = _net.getNrOfNodes();				
			}
		}			
	}
	
	/* Verifica se entre os filhos (children) de um node ha nodes que estao antes dele na 
	 * ordenacao e assim poderiam ser seus pais. Neste caso, estes nodes serao removidos 
	 * do vetor de filhos (children). 
	 * */	
	private void verifyChildren(ArrayList<Integer> children, int[] _vetVariable, int classPosition)
	{	
		ArrayList<Integer> childrenAux = (ArrayList<Integer>) children.clone();
		int position;
		
		for(int i=0; i<childrenAux.size(); i++)
		{
			for(int j=0; j<=classPosition; j++)	
			{
				//System.out.println("Filho: " + childrenAux.elementAt(i));
				if(childrenAux.get(i)== _vetVariable[j])
				{
					position = children.indexOf(childrenAux.get(i));
					children.remove(position);
					j = classPosition+1;
				}
			}			
		}		
	}
	
	private void saveFileNet(EditableBayesNet net, int i, String inputFileName/*, double g*/) 
			throws IOException
	{
		// Nome do arquivo a ser salvo.
		String outputName = "DMBC_";				
		outputName = outputName.concat(String.valueOf(i));
		outputName = outputName.concat("_");
		String outputFileName = inputFileName.replace(".arff", ".xml");
		outputFileName = outputName.concat(outputFileName);
		// Salva em arquivo a rede bayesiana aprendida pelo DMBC.
	    FileWriter fileWriter = new FileWriter(outputFileName);
		PrintWriter fileNet = new PrintWriter(fileWriter, true);
		fileNet.write(net.toXMLBIF03());
		fileNet.close();
		// guarda o valor da funcao g em arquivo
		/*FileWriter fileG = new FileWriter("File_GValue.txt", true);
		PrintWriter fw = new PrintWriter(fileG, true);
		fw.print(g);
		//fw.println(runTime);
		fw.close();*/
	}
	
	private void initVetPosition(int vet[])
	{
		for(int i=0; i<vet.length; i++)
		{
			vet[i] = -1;
		}
	}
	
	private boolean isPresent(int vet[], int variable)
	{
		for(int i=0; i<vet.length; i++)
		{
			if(vet[i] == variable)
				return true;
		}
		return false;
	}
	
	private void saveNet(EditableBayesNet bayesianNet, 
			             String inputFileName, 
			             double _runTime)
	{		
		try 
		{
			// Nome do arquivo a ser salvo.
			String outputName = "Unrestricted-DMBC_" + minWeight + "_";				
			String outputFileName = inputFileName.replace(".arff", ".xml");
			outputFileName = outputName.concat(outputFileName);
			// Salva em arquivo a rede bayesiana aprendida pelo DMBC.
		    FileWriter fileWriter = new FileWriter(outputFileName);
			PrintWriter fileNet = new PrintWriter(fileWriter, true);
			fileNet.write(bayesianNet.toXMLBIF03());
			fileNet.close();
			
			// salva o valor do tempo necessario para a execucao.
			outputName = "Time_Unrestricted-DMBC_";
			outputFileName = inputFileName.replace(".arff", ".txt");
			outputFileName = outputName.concat(outputFileName);
			FileWriter file = new FileWriter(outputFileName, true);
			PrintWriter fw = new PrintWriter(file, true);
			fw.println(_runTime);
			fw.close();
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void showNet(String bayesianNet, double value)
	{		
		try 
		{
			System.out.println("\n\nRede Bayesiana em XML: ");			
			System.out.println(bayesianNet);
			System.out.println("Valor g: " + value);
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void trainAndTestSplitClassification(EditableBayesNet _bayesianNet)
	{
		try
		{
			int seed=1;
			Random rnd = new Random(seed);
			instances.randomize(rnd);
			double percent = 66.0;
	

			instances.setClassIndex(indexclassVariable);
			//System.out.println(instances.classIndex());
			System.out.println("\nPerforming " + percent +"% split evaluation");
	
			int trainSize = (int) Math.round(instances.numInstances()*percent/100);
	
			int testSize = instances.numInstances()-trainSize;
	
	
			Instances train = new Instances (instances, 0, trainSize);
			Instances test = new Instances (instances, trainSize,testSize);
			
			// build and evaluate classifier
			BayesNet netCopy = (BayesNet) BayesNet.makeCopy(_bayesianNet);
			// Classifier clsCopy = Classifier.makeCopy(cls);
			//netCopy.m_Instances = train;
			//netCopy.initStructure();
			//netCopy.buildStructure();
			//netCopy.estimateCPTs();
			netCopy.buildClassifier(train);
			eval = new Evaluation(train);
			eval.evaluateModel(netCopy, test);
			
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString("\n=== Confusion matrix ===\n"));
		}
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}