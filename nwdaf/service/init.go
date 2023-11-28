package service

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	// for subscribtion
	"encoding/json"
	"encoding/csv"
	"io/ioutil"
	"net/http"
	"time"
	// self-defined
	"github.com/free5gc/http2_util"
	"github.com/free5gc/logger_util"
	"github.com/free5gc/path_util"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
	"nwdaf.com/anlf"
	"nwdaf.com/consumer"
	nwdaf_context "nwdaf.com/context"
	"nwdaf.com/factory"
	"nwdaf.com/logger"
	"nwdaf.com/mtlf"
	"nwdaf.com/util"
)

type NWDAF struct{}

type (
	// config information
	Config struct {
		nwdafcfg string
	}
)

var config Config

var nwdafCLi = []cli.Flag{
	cli.StringFlag{
		Name:  "free5gccfg",
		Usage: "common config file",
	},
	cli.StringFlag{
		Name:  "nwdafcfg",
		Usage: "nwdaf config file",
	},
}

var initLog *logrus.Entry

func init() {
	initLog = logger.InitLog
}

func (*NWDAF) GetCliCmd() (flags []cli.Flag) {
	return nwdafCLi
}

func (nwdaf *NWDAF) Initialize(c *cli.Context) error {
	config = Config{
		nwdafcfg: c.String("nwdafcfg"),
	}

	if config.nwdafcfg != "" {
		if err := factory.InitConfigFactory(config.nwdafcfg); err != nil {
			return err
		}
	} else {
		DefaultNWDAFConfigPath := path_util.Free5gcPath("~/free5gc/config/nwdafcfg.yaml")
		if err := factory.InitConfigFactory(DefaultNWDAFConfigPath); err != nil {
			return err
		}
	}

	nwdaf.setLogLevel()

	if err := factory.CheckConfigVersion(); err != nil {
		return err
	}

	return nil
}

func (nwdaf *NWDAF) setLogLevel() {
	if factory.NwdafConfig.Logger == nil {
		initLog.Warnln("NWDAF config without log level setting!!!")
		return
	}

	logger.SetLogLevel(logrus.InfoLevel)

}

func (nwdaf *NWDAF) FilterCli(c *cli.Context) (args []string) {
	for _, flag := range nwdaf.GetCliCmd() {
		name := flag.GetName()
		value := fmt.Sprint(c.Generic(name))
		if value == "" {
			continue
		}

		args = append(args, "--"+name, value)
	}
	return args
}

func (nwdaf *NWDAF) Exec(c *cli.Context) error {

	initLog.Traceln("args:", c.String("nwdafcfg"))
	args := nwdaf.FilterCli(c)
	initLog.Traceln("filter: ", args)
	command := exec.Command("./nwdaf", args...)

	wg := sync.WaitGroup{}
	wg.Add(3)

	stdout, err := command.StdoutPipe()
	if err != nil {
		initLog.Fatalln(err)
	}
	go func() {
		in := bufio.NewScanner(stdout)
		for in.Scan() {
			fmt.Println(in.Text())
		}
		wg.Done()
	}()

	stderr, err := command.StderrPipe()
	if err != nil {
		initLog.Fatalln(err)
	}
	go func() {
		in := bufio.NewScanner(stderr)
		for in.Scan() {
			fmt.Println(in.Text())
		}
		wg.Done()
	}()

	go func() {
		if errCom := command.Start(); errCom != nil {
			initLog.Errorf("NWDAF start error: %v", errCom)
		}
		wg.Done()
	}()

	wg.Wait()

	return err
}

// ========================
// == subscribes amf/oam ==
// ========================

// type define below could be import from amf/contest.. TODO
type PduSessions struct {
	PduSessionId string `json:"PduSessionId"`
	SmContextRef string `json:"smContextRef"`
	Sst string `json:"Sst"`
	Sd string `json:"Sd"`
	Dnn string`json:"Dnn"`
}

type ue_context struct{
	AccessType string `json:"AccessType"`
	Supi string `json:"Supi"`
	Guti string `json:"Guti"`
	Mcc string `json:"Mcc"`
	Mnc string `json:"Mnc"`
	Tac string `json:"Tac"`

	// nested decoding TODO
	/*
	PduSessions struct {
		PduSessionId string `json:"PduSessionId"`
		SmContextRef string `json:PduSessions.SmContextRef`
		Sst string `json:PduSessions.Sst`
		Sd string `json:PduSessions.Sd`
		Dnn string `json:"PduSessions.Dnn"`
	} `json:"PduSessions"` // this contains more sub-classes. just for simple usage here 
	*/

	PduSessions []PduSessions `json:"PduSessions"`
	CmState string `json:"CmState"`
	now_time string
}

//(nwdaf *NWDAF)
func (nwdaf *NWDAF) Subscribe_amf_oam() {
	
	initLog.Infoln("Periodically data collecting")
	//initLog.Infoln("Subscribtion")
	// ===
	// when server is up. it is default to http2
	// so http.get is http2 
	// needed to add line to force HTTP2 to false 
	// using http only
	transport := &http.Transport{
		ForceAttemptHTTP2: false,
	}
	http := &http.Client{Transport: transport}
	// ===
	
	// subscribe for amf oam get_registered_ue_context for now.
	url := "http://127.0.0.18:8000/namf-oam/v1/registered-ue-context"
	// the url is just for local VM IP should be modify in k8s.
	resp, err := http.Get(url)
	if err != nil{
		//panic(err)
		//println("error happends while getting registered UE context")
		return 
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	var jsonBlob = []byte(body)

	var contexts []ue_context
	json.Unmarshal(jsonBlob, &contexts)
	//println(contexts[0].PduSessions[0].PduSessionId)
	
	readCSV, err := os.Open("data.csv") 
	// will save to free5gc folders
	if err != nil{
		println("no data.csv exists")
		
		// first subscribe write to .csv
		outputCSV, err := os.Create("data.csv")
		if err != nil{return }
		defer readCSV.Close()
		writer := csv.NewWriter(outputCSV)
    		defer writer.Flush()

		now_t := time.Now().Format("2006-01-02 15:04:05")
		for _, UE := range contexts {
			var csvRow []string
			UE.now_time = now_t
			csvRow = append(
				csvRow,
				UE.AccessType,
				UE.Supi,
				UE.Guti,
				UE.Mcc,
				UE.Mnc,
				UE.Tac,
				// PduSessions
				UE.PduSessions[0].PduSessionId,
				UE.PduSessions[0].SmContextRef,
				UE.PduSessions[0].Sst,
				UE.PduSessions[0].Sd,
				UE.PduSessions[0].Dnn,
				UE.now_time)
			println("UE_record Supi: ", UE.Supi) // debug 
			if err := writer.Write(csvRow); err != nil {return}
			
		}
	}

	// not first time
	records, err := csv.NewReader(readCSV).ReadAll()
	if err != nil{ println("reader failed")
	}
	for _, UE_record := range records {
		var r_csvRow ue_context
		//println("UE_record Supi: ", UE_record[2])
		r_csvRow.AccessType = UE_record[0]
		r_csvRow.Supi = UE_record[1]     //UE.Supi,
		r_csvRow.Guti = UE_record[2]     //UE.Guti,
		r_csvRow.Mcc = UE_record[3]      //UE.Mcc,
		r_csvRow.Mnc = UE_record[4]      //UE.Mnc,
		r_csvRow.Tac = UE_record[5]      //]UE.Tac,
		r_csvRow.now_time = UE_record[6] //UE.now_time)
		//if err := writer.Write(r_csvRow); err != nil{return}
		contexts = append(contexts, r_csvRow)
	}

	now_t := time.Now().Format("2006-01-02 15:04:05")
	//println("current curl time is ", now_t)

	outputCSV, err := os.Create("data.csv")
	writer := csv.NewWriter(outputCSV)
	//defer writer.Flush()

	for _, UE := range contexts {
		var csvRow []string
		if (UE.now_time == "") {
			UE.now_time = now_t
		}
		csvRow = append(
			csvRow,
			UE.AccessType,
			UE.Supi,
			UE.Guti,
			UE.Mcc,
			UE.Mnc,
			UE.Tac,
			UE.now_time)
		if err := writer.Write(csvRow); err != nil {return}
	}
	writer.Flush()
} 



func (nwdaf *NWDAF) Start() {

	initLog.Infoln("Server started")

	if !util.InitNWDAFContext() {
		initLog.Error("Initicating context failed")
		return
	}

	wg := sync.WaitGroup{}

	self := nwdaf_context.NWDAF_Self()
	util.InitNwdafContext(self)

	addr := fmt.Sprintf("127.0.0.1:24242")
	router := logger_util.NewGinWithLogrus(logger.GinLog)
	mtlf.AddService(router)
	anlf.AddService(router)

	profile := consumer.BuildNFInstance(self)
	var newNrfUri string
	var err error

	newNrfUri, self.NfId, err = consumer.SendRegisterNFInstance(self.NrfUri, profile.NfInstanceId, profile)
	if err == nil {
		self.NrfUri = newNrfUri
	} else {
		initLog.Errorf("Send Register NFInstance Error[%s]", err.Error())
	}

	// create a sigalChannel
	signalChannel := make(chan os.Signal, 1)
	signal.Notify(signalChannel, os.Interrupt, syscall.SIGTERM)
	
	// ticker to trigger amf subscribtion
	//ticker := time.NewTicker(5 * time.Second)
	go func() {
		// channel reciever
		//<-signalChannel
		//os.Exit(0)
		for{
			select {
				//case <- ticker.C:
					//Subscribe_amf_oam()
					//os.Exit(0)
				case <-signalChannel: // ctrl-C
					//ticker.Stop()
					os.Exit(0)
			}
		}
	}()
	
	server, err := http2_util.NewServer(addr, "nwdafsslkey.log", router)
	if server == nil {
		initLog.Errorf("Initialize HTTP server failed: %+v", err)
		return
	}
	if err != nil {
		initLog.Warnln("Initialize HTTP server:", err)
	}
	serverScheme := factory.NwdafConfig.Configuration.Sbi.Scheme
	if serverScheme == "http" {
		err = server.ListenAndServe()
	} else if serverScheme == "https" {
		err = server.ListenAndServe() //TODO: changing to HTTPS (TLS)
	}

	if err != nil {
		initLog.Fatalln("HTTP server setup failed:", err)
	}
	initLog.Info("NWDAF running...")

	wg.Wait()
}