/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 * K2.java
 * Copyright (C) 2001 University of Waikato, Hamilton, New Zealand
 * 
 */
package classifiers.bayes.net.search.local;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;



/**
 <!-- globalinfo-start -->
 * This Bayes Network learning algorithm uses a hill climbing algorithm restricted by an order on the variables.<br/>
 * <br/>
 * For more information see:<br/>
 * <br/>
 * G.F. Cooper, E. Herskovits (1990). A Bayesian method for constructing Bayesian belief networks from databases.<br/>
 * <br/>
 * G. Cooper, E. Herskovits (1992). A Bayesian method for the induction of probabilistic networks from data. Machine Learning. 9(4):309-347.<br/>
 * <br/>
 * Works with nominal variables and no missing values only.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;proceedings{Cooper1990,
 *    author = {G.F. Cooper and E. Herskovits},
 *    booktitle = {Proceedings of the Conference on Uncertainty in AI},
 *    pages = {86-94},
 *    title = {A Bayesian method for constructing Bayesian belief networks from databases},
 *    year = {1990}
 * }
 * 
 * &#64;article{Cooper1992,
 *    author = {G. Cooper and E. Herskovits},
 *    journal = {Machine Learning},
 *    number = {4},
 *    pages = {309-347},
 *    title = {A Bayesian method for the induction of probabilistic networks from data},
 *    volume = {9},
 *    year = {1992}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -N
 *  Initial structure is empty (instead of Naive Bayes)</pre>
 * 
 * <pre> -P &lt;nr of parents&gt;
 *  Maximum number of parents</pre>
 * 
 * <pre> -R
 *  Random order.
 *  (default false)</pre>
 * 
 * <pre> -mbc
 *  Applies a Markov Blanket correction to the network structure, 
 *  after a network structure is learned. This ensures that all 
 *  nodes in the network are part of the Markov blanket of the 
 *  classifier node.</pre>
 * 
 * <pre> -S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]
 *  Score type (BAYES, BDeu, MDL, ENTROPY and AIC)</pre>
 * 
 <!-- options-end -->
 *
 * @author Remco Bouckaert (rrb@xm.co.nz)
 * @version $Revision: 1.8 $
 */
public class DMBC 
 	extends LocalScoreSearchAlgorithm
 	implements TechnicalInformationHandler {
  
  	/** for serialization */
  	static final long serialVersionUID = 6176545934752116631L;
  	private int gFunctionCounter = 0;
  
	/** Holds flag to indicate ordering should be random **/
	boolean m_bRandomOrder = false;

	/** Holds variable ordering **/
	int variableOrdering[];

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
	  TechnicalInformation 	result;
	  TechnicalInformation 	additional;
	  
	  result = new TechnicalInformation(Type.PROCEEDINGS);
	  result.setValue(Field.AUTHOR, "G.F. Cooper and E. Herskovits");
	  result.setValue(Field.YEAR, "1990");
	  result.setValue(Field.TITLE, "A Bayesian method for constructing Bayesian belief networks from databases");
	  result.setValue(Field.BOOKTITLE, "Proceedings of the Conference on Uncertainty in AI");
	  result.setValue(Field.PAGES, "86-94");
	  
	  additional = result.add(Type.ARTICLE);
	  additional.setValue(Field.AUTHOR, "G. Cooper and E. Herskovits");
	  additional.setValue(Field.YEAR, "1992");
	  additional.setValue(Field.TITLE, "A Bayesian method for the induction of probabilistic networks from data");
	  additional.setValue(Field.JOURNAL, "Machine Learning");
	  additional.setValue(Field.VOLUME, "9");
	  additional.setValue(Field.NUMBER, "4");
	  additional.setValue(Field.PAGES, "309-347");
	  
	  return result;
	}

	/**
	 * search determines the network structure/graph of the network
	 * with the DMBC algorithm, restricted by its initial structure (which can
	 * be an empty graph, or a Naive Bayes graph.
	 * 
	 * @param bayesNet the network
	 * @param instances the data to work with
	 * @throws Exception if something goes wrong
	 */
	public void search (BayesNet bayesNet, Instances instances) throws Exception {
		int nOrder[] = new int [instances.numAttributes()];
		int classAttribute = instances.classIndex();
		nOrder[0] = classAttribute;

		int nAttribute = 0;

		if(variableOrdering==null)
		{
			for (int iOrder = 1; iOrder < instances.numAttributes(); iOrder++) {
			  if (nAttribute == instances.classIndex()) {
			    nAttribute++;
			  } 
			  nOrder[iOrder] = nAttribute++;
			} 
		}
		else nOrder = variableOrdering;

		if (m_bRandomOrder) {
			// generate random ordering (if required)
			Random random = new Random();
					int iClass = 0;
					/*
					if (getInitAsNaiveBayes()) {
						iClass = 0; 
					} else {
						iClass = -1;
					}
					*/
			int iOrder = 0;
			nOrder[iOrder] = instances.classIndex();
			iOrder++;
			for (; iOrder < (instances.numAttributes()-1); iOrder++) {
			int iOrder2 = Math.abs(random.nextInt()) % instances.numAttributes();
						if (iOrder != iClass && iOrder2 != iClass) {
							int nTmp = nOrder[iOrder];
							nOrder[iOrder] = nOrder[iOrder2];
							nOrder[iOrder2] = nTmp;
						}
			}
		}

		// determine base scores
		double [] fBaseScores = new double [instances.numAttributes()];
		for (int iOrder = 0; iOrder < instances.numAttributes(); iOrder++) {
			int iAttribute = nOrder[iOrder];
			fBaseScores[iAttribute] = calcNodeScore(iAttribute);
		}
		
		// DMBC algorithm: greedy search restricted by ordering
		// Itera atraves dos atributos da instancia, comecando do segundo atributo (1) ate o penultimo.
		for (int iOrder = 1; iOrder < instances.numAttributes(); iOrder++) {
		    // Cria um novo conjunto de pais vazio para o atributo atual.
		    ParentSet parentSet = new ParentSet();
		    // Obtem o atributo atual.
		    int iAttribute = nOrder[iOrder];
		    // Inicializa o melhor score com o score base do atributo.
		    double fBestScore = fBaseScores[iAttribute];

		    // Verifica se e possivel adicionar mais pais ao atributo atual.
		    boolean bProgress = (bayesNet.getParentSet(iAttribute).getNrOfParents() < getMaxNrOfParents());
		    
		    // Enquanto for possivel adicionar mais pais.
		    while (bProgress) {
		        // Inicializa o atributo que sera considerado o melhor pai.
		        int nBestAttribute = -1;

		        // Percorre todos os atributos anteriores ao atributo atual.
//		        for (int iOrder2 = 0; iOrder2 < iOrder; iOrder2++) {
		        for (int iOrder2 = 0; iOrder2 < instances.numAttributes(); iOrder2++) {		//modificado -> compara com todos
		        	if (iOrder2 != iOrder) {
		        		// Obtem o atributo
			            int iAttribute2 = nOrder[iOrder2];
			            // Calcula um score com a adicao desse atributo como pai do atributo atual.
			            double fScore = calcScoreWithExtraParent(iAttribute, iAttribute2);
			            // Incrementa um contador de funcao.
			            gFunctionCounter++;
			            
			            // Se o score calculado for melhor que o melhor score ate agora.
//			            if (fScore > fBestScore) {
		            	if (fScore > fBestScore || (fScore == fBestScore && iAttribute2 < nBestAttribute)) {	//modificado -> criterio de desempate
			                // Verifica se o atributo anterior eh a classe ou eh um filho da classe.
			                if((iAttribute2 == classAttribute) || (bayesNet.getParentSet(iAttribute).contains(classAttribute))) {
			                    // Atualiza o melhor score e o atributo considerado o melhor pai.
			                    fBestScore = fScore;
			                    nBestAttribute = iAttribute2;
			                }
			            }
		        	}
		        }
		        
		        
		        // Se foi encontrado um atributo para adicionar como pai.
		        if (nBestAttribute != -1) {
		            // Adiciona o atributo como pai ao conjunto de pais do atributo atual.
		            bayesNet.getParentSet(iAttribute).addParent(nBestAttribute, instances);
		            // Atualiza o score base do atributo.
		            fBaseScores[iAttribute] = fBestScore;
		            // Verifica se ainda eh possivel adicionar mais pais.
		            bProgress = (bayesNet.getParentSet(iAttribute).getNrOfParents() < getMaxNrOfParents());
		        } else {
		            // Se nao foi encontrado um pai adequado, encerra o loop.
		            bProgress = false;
		        }
		        
		    }
		}

	} // buildStructure
	
/**
 *************************************************************
 * 
 */	

	/**
	 * Sets the max number of parents
	 *
	 * @param nMaxNrOfParents the max number of parents
	 */
	public void setMaxNrOfParents(int nMaxNrOfParents) {
	  m_nMaxNrOfParents = nMaxNrOfParents;
	} 

	/**
	 * Gets the max number of parents.
	 *
	 * @return the max number of parents
	 */
	public int getMaxNrOfParents() {
	  return m_nMaxNrOfParents;
	} 

	/**
	 * Sets whether to init as naive bayes
	 *
	 * @param bInitAsNaiveBayes whether to init as naive bayes
	 */
	public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
	  m_bInitAsNaiveBayes = bInitAsNaiveBayes;
	} 
	
	/**
	 * Sets the variable ordering
	 *
	 * @param _variableOrdering whether to init with a variable ordering.
	 */
	public void setVariableOrdering(int[] _variableOrdering) {
		variableOrdering = _variableOrdering;
	} 

	/**
	 * Gets whether to init as naive bayes
	 *
	 * @return whether to init as naive bayes
	 */
	public boolean getInitAsNaiveBayes() {
	  return m_bInitAsNaiveBayes;
	} 

	/** 
	 * Set random order flag 
	 *
	 * @param bRandomOrder the random order flag
	 */
	public void setRandomOrder(boolean bRandomOrder) {
		m_bRandomOrder = bRandomOrder;
	} // SetRandomOrder

	/** 
	 * Get random order flag 
	 *
	 * @return the random order flag
	 */
	public boolean getRandomOrder() {
		return m_bRandomOrder;
	} // getRandomOrder
  
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration listOptions() {
	  Vector newVector = new Vector(0);

	  newVector.addElement(new Option("\tInitial structure is empty (instead of Naive Bayes)", 
					 "N", 0, "-N"));

	  newVector.addElement(new Option("\tMaximum number of parents", "P", 1, 
						"-P <nr of parents>"));

	  newVector.addElement(new Option(
			"\tRandom order.\n"
			+ "\t(default false)",
			"R", 0, "-R"));

		Enumeration enu = super.listOptions();
		while (enu.hasMoreElements()) {
	    	newVector.addElement(enu.nextElement());
		}
	  return newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 *
	 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -N
	 *  Initial structure is empty (instead of Naive Bayes)</pre>
	 * 
	 * <pre> -P &lt;nr of parents&gt;
	 *  Maximum number of parents</pre>
	 * 
	 * <pre> -R
	 *  Random order.
	 *  (default false)</pre>
	 * 
	 * <pre> -mbc
	 *  Applies a Markov Blanket correction to the network structure, 
	 *  after a network structure is learned. This ensures that all 
	 *  nodes in the network are part of the Markov blanket of the 
	 *  classifier node.</pre>
	 * 
	 * <pre> -S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]
	 *  Score type (BAYES, BDeu, MDL, ENTROPY and AIC)</pre>
	 * 
	 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
    
	  setRandomOrder(Utils.getFlag('R', options));

	  m_bInitAsNaiveBayes = !(Utils.getFlag('N', options));

	  String sMaxNrOfParents = Utils.getOption('P', options);

	  if (sMaxNrOfParents.length() != 0) {
		setMaxNrOfParents(Integer.parseInt(sMaxNrOfParents));
	  } else {
		setMaxNrOfParents(100000);
	  }
	  super.setOptions(options);
	}

	/**
	 * Gets the current settings of the search algorithm.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	public String [] getOptions() {
          String[] superOptions = super.getOptions();
	  String [] options  = new String [4 + superOptions.length];
	  int current = 0;
	  options[current++] = "-P";
	  options[current++] = "" + m_nMaxNrOfParents;
	  if (!m_bInitAsNaiveBayes) {
		options[current++] = "-N";
	  }	  if (getRandomOrder()) {
		options[current++] = "-R";
	  }

          // insert options from parent class
          for (int iOption = 0; iOption < superOptions.length; iOption++) {
                  options[current++] = superOptions[iOption];
          }

	  while (current < options.length) {
		options[current++] = "";
	  }
	  // Fill up rest with empty strings, not nulls!
	  return options;
	}

	/**
	 * This will return a string describing the search algorithm.
	 * @return The string.
	 */
	public String globalInfo() {
	  return
	      "This Bayes Network learning algorithm uses a hill climbing algorithm "
	    + "restricted by an order on the variables.\n\n"
	    + "For more information see:\n\n"
	    + getTechnicalInformation().toString() + "\n\n"
	    + "Works with nominal variables and no missing values only.";
	}

	/**
	 * @return a string to describe the RandomOrder option.
	 */
	public String randomOrderTipText() {
	  return "When set to true, the order of the nodes in the network is random." +
	  " Default random order is false and the order" +
	  " of the nodes in the dataset is used." +
	  " In any case, when the network was initialized as Naive Bayes Network, the" +
	  " class variable is first in the ordering though.";
	} // randomOrderTipText

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
	  return RevisionUtils.extract("$Revision: 1.8 $");
	}
	
	public int getGFunctionCounter() {
		return gFunctionCounter;
	}

	public void setGFunctionCounter(int gFunctionCounter) {
		this.gFunctionCounter = gFunctionCounter;
	}
}