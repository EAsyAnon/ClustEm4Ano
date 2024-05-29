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
import org.deidentifier.arx.criteria.TCloseness;

/**
 * This class implements an example on how to use the API by providing CSV files
 * as input.
 *
 * @author Fabian Prasser
 * @author Florian Kohlmayer
 */
public class Baseline extends Example {
    
    /**
     * Entry point.
     * 
     * @param args the arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
    	
        int[] kValues = {2,5,10,15,20,25,30,50,100,150,200};
                
        for (int k : kValues) {
        	
        	// original data
            Data data = Data.create("./input/baseline/adult.data.csv", StandardCharsets.UTF_8, ',');
            
            String dataset_name = "adult";
       
            
            System.out.println("k= " + String.valueOf(k));
                                    
            
            data.getDefinition().setAttributeType("workclass", Hierarchy.create("./input/baseline/adult_hierarchy_workclass.csv", StandardCharsets.UTF_8, ';'));
            data.getDefinition().setAttributeType("workclass", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

            data.getDefinition().setAttributeType("education", Hierarchy.create("./input/baseline/adult_hierarchy_education.csv", StandardCharsets.UTF_8, ';'));
            data.getDefinition().setAttributeType("education", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

            data.getDefinition().setAttributeType("native-country", Hierarchy.create("./input/baseline/adult_hierarchy_native-country.csv", StandardCharsets.UTF_8, ';'));
            data.getDefinition().setAttributeType("native-country", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
            
            data.getDefinition().setAttributeType("occupation", Hierarchy.create("./input/baseline/adult_hierarchy_occupation.csv", StandardCharsets.UTF_8, ';'));
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
	        
	        // Write results
	        System.out.println(" - Writing data...");
	        
            String fileName = "./experiment_0/anonymized_l=2_k=" + String.valueOf(k) + ".csv";
	        result.getOutput(false).save(fileName, ';');
	        
	        System.out.println("Done!");
        }
    }
}