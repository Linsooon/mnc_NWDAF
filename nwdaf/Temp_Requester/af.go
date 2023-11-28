package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

func requestModelTraining(reqNfInstanceId string) { //TODO: Change input data 'data' to appropriate attribute
	jsonBody := map[string]interface{}{}
	jsonBody["reqNFInstanceID"] = reqNfInstanceId
	jsonBody["nfService"] = "training"
	now_t := time.Now().Format("2006-01-02 15:04:05")
	jsonBody["reqTime"] = now_t
	jsonBody["data"] = "none"
	jsonStr, _ := json.Marshal(jsonBody)
	print("*********")
	resp, err := http.Post("http://localhost:24242/nwdaf-mtlf/v1/:training", "application/json", bytes.NewBuffer([]byte(jsonStr)))
	if err != nil {
		fmt.Println("error: %v", err)
	} else {
		// posting model training request succeed.
		fmt.Println("")
		fmt.Println("=======================")
		fmt.Println("===== MTLF Result =====")
		fmt.Println("=======================")
		fmt.Println("")
		fmt.Println("== Responese header ==")
		fmt.Println(resp.Header)
		fmt.Println("=======================")
		fmt.Println("")
		respBody, _ := ioutil.ReadAll(resp.Body)
		jsonData := map[string]interface{}{}
		json.Unmarshal(respBody, &jsonData)
		fmt.Println("== Responese data ==")
		fmt.Println(jsonData)
		fmt.Println("=======================")
	}

}

func requestModelInference(reqNfInstanceId string){ //data_num string) { //TODO: Change input data 'data' to appropriate attribute
	jsonBody := map[string]interface{}{}
	jsonBody["reqNFInstanceID"] = reqNfInstanceId
	jsonBody["nfService"] = "inference"
	now_t := time.Now().Format("2006-01-02 15:04:05")
	jsonBody["reqTime"] = now_t
	//jsonBody["data"] = data_num
	jsonStr, _ := json.Marshal(jsonBody)
	print("*********")
	resp, err := http.Post("http://localhost:24242/nwdaf-anlf/v1/:inference", "application/json", bytes.NewBuffer([]byte(jsonStr)))
	if err != nil {
		fmt.Println("error: %v", err)
	} else {
		fmt.Println("")
		fmt.Println("=======================")
		fmt.Println("===== AnLF Result =====")
		fmt.Println("=======================")
		fmt.Println("")
		fmt.Println("== Responese header ==")
		fmt.Println("=======================")
		fmt.Println(resp.Header)
		fmt.Println("")
		respBody, _ := ioutil.ReadAll(resp.Body)
		jsonData := map[string]interface{}{}
		json.Unmarshal(respBody, &jsonData)
		fmt.Println("=== Result below ===")
		fmt.Println(jsonData)
		fmt.Println("=======================")
	}

}

func main() {
	n := 1
	var selection int
	//var data_num string
	for n < 10 {
		fmt.Println("Choose the function - 1) MTLF, 2) AnLF :")
		fmt.Scanln(&selection)
		if selection == 1 {
			requestModelTraining("Model-training")
		} else if selection == 2 {
			//fmt.Println("Choose the data number for prediction:")
			//fmt.Scanln(&data_num)
			requestModelInference("Test NF-Function")
		} else {
			fmt.Println("Wrong number")
		}
	}

}
