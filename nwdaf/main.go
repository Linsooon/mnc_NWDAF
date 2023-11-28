package main

import (
	"fmt"
	"os"
	"time"
	//"sync"
	
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"

	"github.com/free5gc/version"
	"nwdaf.com/logger"
	"nwdaf.com/service"
)

// func main() {
// 	message := service.Hello("servicehello")
// 	fmt.Println(message)
// 	message2 := logger.Hello("loggerhello")
// 	fmt.Println(message2)
// 	message3 := factory.Hello("factoryhello")
// 	fmt.Println(message3)

// }

var NWDAF = &service.NWDAF{}

var appLog *logrus.Entry

func init() {
	appLog = logger.AppLog
}

func main() {
	app := cli.NewApp()
	app.Name = "nwdaf"
	appLog.Infoln(app.Name)
	appLog.Infoln("NWDAF version: ", version.GetVersion())
	app.Usage = "-free5gccfg common configuration file -nwdafcfg nwdaf configuration file"
	app.Action = action
	app.Flags = NWDAF.GetCliCmd()
	if err := app.Run(os.Args); err != nil {
		appLog.Errorf("NWDAF Run error: %v", err)
		return
	}
	// === Subscribe ===
	/*
	sub_app := cli.NewApp()
	sub_app.Name = "nwdaf_subscription"
	appLog.Infoln(sub_app.Name)
	sub_app.Action = Subscribe
	if err := sub_app.Run(os.Args); err != nil {
		appLog.Errorf("NWDAF Run error: %v", err)
		return
	}
	*/
}


func action(c *cli.Context) error {
	if err := NWDAF.Initialize(c); err != nil {
		logger.CfgLog.Errorf("%+v", err)
		return fmt.Errorf("Failed to initialize !!")
	}
	
	// ticker to trigger amf subscribtion
	//ticker := time.NewTicker(5 * time.Second)
	
	NWDAF.Subscribe_amf_oam() // 成功
	go NWDAF.Start()
	
	for{
		time.Sleep(10 * time.Second)
		NWDAF.Subscribe_amf_oam()
	}
	
	//var wait_g sync.WaitGroup
	
	/*
	go func() {
		// channel reciever
		//<-signalChannel
		//os.Exit(0)
		for{
			//wait_g.Add(1)
			//time.Sleep(5 * time.Second)
			//NWDAF.Subscribe_amf_oam() 
			select {
				case <- ticker.C:
					//defer wait_g.Done()
					NWDAF.Subscribe_amf_oam() // 失敗
				
				//case <-signalChannel:
				//	ticker.Stop()
				//	os.Exit(0)
				
			}
			
		}
	}()
	*/
	//wait_g.Wait()
	
	return nil
}
/*
func Subscribe(){
	NWDAF.Subscribe()
	return 
}
*/