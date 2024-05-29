/*
 * ARX Data Anonymization Tool
 * Copyright 2012 - 2023 Fabian Prasser and contributors
 * 
 * Modifications copyright (C) 2024 Technische Hochschule Deggendorf / Robert Aufschläger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

package org.deidentifier.arx.examples;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.criteria.DDisclosurePrivacy;
import org.deidentifier.arx.criteria.DistinctLDiversity;
import org.deidentifier.arx.criteria.EqualDistanceTCloseness;
import org.deidentifier.arx.criteria.KAnonymity;
import java.io.File;

/**
 * This class implements an example on how to use the API by providing CSV files
 * as input.
 *
 * @author Fabian Prasser
 * @author Florian Kohlmayer
 */
public class Anonymize extends Example {
	
    public static String findFileWithPrefix(String folderPath, String prefix) {
        File folder = new File(folderPath);
        File[] files = folder.listFiles();

        if (files != null) {
            for (File file : files) {
                if (file.isFile() && file.getName().startsWith(prefix)) {
                    return file.getName();
                }
            }
        }
        return null; // Return null if no matching file is found
    }
    
    /**
     * Entry point.
     * 
     * @param args the arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
    	
    	long startTime = System.nanoTime();
    	
    	String[] clusterings = {"agglomerative", "kmeans"};


    	
    	String[] embeddings = {  
    			"BERT", "word2vec", "average_word_embeddings_glove.6B.300d", "msmarco-bert-base-dot-v5",
                "multi-qa-mpnet-base-dot-v1", "text-embedding-3-large", "text-embedding-3-small",
                "mistral-embed", "jinaai-jina-embeddings-v2-base-en", 
                "fasttext",
                "average_word_embeddings_komninos", "average_word_embeddings_levy_dependency",
                "average_word_embeddings_glove.840B.300d"};
    	
        int[] kValues = {2,5,10,15,20,25,30,50,100,150,200};

    	
        for (String clustering : clusterings) {
        	
	        for (String embedding : embeddings) {
	    	             
		        for (int k : kValues) {
		        	
		        	// original data
		            Data data = Data.create("./input/baseline/adult.data.csv", StandardCharsets.UTF_8, ',');
		            
		            String dataset_name = "adult";
		       
		            String folderPath = "./input/" + clustering + "/" + embedding;
		            String prefix = dataset_name + "_" + clustering + "_" + embedding + "_";
		            
		            System.out.println(prefix + "k=" + String.valueOf(k));
		            
		            // workclass
		            String prefix_workclass = prefix + "workclass_vgh";
		            String fileNameWorkclassVgh = findFileWithPrefix(folderPath, prefix_workclass);
		            data.getDefinition().setAttributeType("workclass", Hierarchy.create(folderPath + "/" + fileNameWorkclassVgh, StandardCharsets.UTF_8, ','));
		            data.getDefinition().setAttributeType("workclass", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		            
		            // education
		            String prefix_education = prefix + "education_vgh";
		            String fileNameEducationVgh = findFileWithPrefix(folderPath, prefix_education);
		            data.getDefinition().setAttributeType("education", Hierarchy.create(folderPath + "/" + fileNameEducationVgh, StandardCharsets.UTF_8, ','));
		            data.getDefinition().setAttributeType("education", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
		            
		            // native-country
		            String prefix_native_country = prefix + "native-country_vgh";
		            String fileNameNativeCountryVgh = findFileWithPrefix(folderPath, prefix_native_country);
		            data.getDefinition().setAttributeType("native-country", Hierarchy.create(folderPath + "/" + fileNameNativeCountryVgh, StandardCharsets.UTF_8, ','));
		            data.getDefinition().setAttributeType("native-country", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

		            // occupation
		            String prefix_occupation = prefix + "occupation_vgh";
		            String fileNameOccupationVgh = findFileWithPrefix(folderPath, prefix_occupation);
		            data.getDefinition().setAttributeType("occupation", Hierarchy.create(folderPath + "/" + fileNameOccupationVgh, StandardCharsets.UTF_8, ','));
		            data.getDefinition().setAttributeType("occupation", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

		            data.getDefinition().setAttributeType("age", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("fnlwgt", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("education-num", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("marital-status", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("race", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("capital-gain", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("capital-loss", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("sex", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("relationship", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("hours-per-week", AttributeType.INSENSITIVE_ATTRIBUTE);
		            data.getDefinition().setAttributeType("race", AttributeType.INSENSITIVE_ATTRIBUTE);

		            data.getDefinition().setAttributeType("income", AttributeType.SENSITIVE_ATTRIBUTE);
		        	
			        // Create an instance of the anonymizer
			        ARXAnonymizer anonymizer = new ARXAnonymizer();
			        
			        // Execute the algorithm
			        ARXConfiguration config = ARXConfiguration.create();
			        config.addPrivacyModel(new KAnonymity(k));
			        config.addPrivacyModel(new DistinctLDiversity("income", 2));
			        config.setSuppressionLimit(0.5d);
			        ARXResult result = anonymizer.anonymize(data, config);
			        
			        // Print info
			        Example.printResult(result, data);
			        
		            String outputFolderPath = "./experiment_0/" + clustering + "/" + embedding + "/";
		            
		            // CREATE FOLDER
		            
		            File directory = new File(outputFolderPath);
		            
			        System.out.println(directory);
		
		            if (!directory.exists()) {
		                if (directory.mkdir()) {
		                    System.out.println("Directory created successfully!");
		                } else {
		                    System.out.println("Failed to create directory.");
		                }
		            } else {
		                System.out.println("Directory already exists.");
		            }
			        
			        // Write results
			        System.out.println(" - Writing data...");
			        
		            String outputFileName = outputFolderPath + prefix + "anonymized_l=2_k=" + String.valueOf(k) + ".csv";
			        result.getOutput(false).save(outputFileName, ';');
			        
			        System.out.println("Done!");
		        }
	        }

        }
        
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        double seconds = TimeUnit.SECONDS.convert(duration, TimeUnit.NANOSECONDS);
        
        System.out.println("Total execution time: " + seconds + " seconds.");
    }
}