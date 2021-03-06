//+------------------------------------------------------------------+
//|                                                 BS CAPITAL.mq5   |
//|                        Copyright 2016, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2016, Vinicius Barbosa Paiva."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>;
CTrade Trade;

#include <Trade\TradeAlt.mqh>;
CTradeAlt TradeAlt;

#include <Trade\Pending.mqh>;
CPending Pending;

#include <Trade\lib_cisnewbar.mqh>;
CisNewBar current_chart; 

#include <Trade\PositionInfo.mqh>;
CPositionInfo  Position;

#include <Trade\AccountInfo.mqh>;
CAccountInfo Account;

#include <Trade\OrderInfo.mqh>;
COrderInfo OrderInfor;

#include <Trade\DealInfo.mqh>;
CDealInfo DealInfor;

#include <ChartObjects\ChartObject.mqh>;
CChartObject ChartObject;

#include <errordescription.mqh>
#include <MQL4MarketInfo.mqh>
#define MAX_RETRIES 5		// Max retries on error
#define RETRY_DELAY 3000	// Retry delay in ms

//--- input parameters
sinput string Setup; // Which setup will use?
input bool titao= true;// Setup Titao?
input bool titao_tese= false;// Setup Tito Tese?
input bool titao_scalpe= false;// Setup Tito Scaper?
input bool nove_tres= false;// Setup 9.3?

sinput string BB; // Bollinger Bands
input int BandsPeriod= 20;
input int BandsShift = 0;
input double BandsDeviation=2;
input ENUM_APPLIED_PRICE BandsPrice=PRICE_CLOSE;

sinput string M20; // High Moving Average 
input int MAPeriod20= 20; // Period High MA 
input int MAShift20 = 0; // Shift High MA
input ENUM_MA_METHOD MAMethod20=MODE_SMA; // Method High MA
input ENUM_TIMEFRAMES MATime20 = PERIOD_M2; // TimeFrame High MA
input ENUM_APPLIED_PRICE MAPrice20=PRICE_HIGH; // Applied Price High MA

sinput string M17; // Low Moving Average 
input int MAPeriod17= 20; // Period Low MA
input int MAShift17 = 0; // Shift Low MA
input ENUM_MA_METHOD MAMethod17=MODE_SMA; // Method Low MA
input ENUM_TIMEFRAMES MATime17 = PERIOD_M2; // TimeFrame Low MA
input ENUM_APPLIED_PRICE MAPrice17=PRICE_LOW; // Applied Price Low MA

sinput string M34; // Activation Moving Average 
input int MAPeriod34= 9; // Period Activation MA
input int MAShift34 = 0; // Shift Activation MA
input ENUM_MA_METHOD MAMethod34=MODE_EMA; // Method Activation MA
input ENUM_TIMEFRAMES MATime34 = PERIOD_M2; // TimeFrame Activation MA
input ENUM_APPLIED_PRICE MAPrice34=PRICE_CLOSE; // Applied Price Activation MA

sinput string VolumeEa; // Define Volume
input double Volume=1;
sinput string Stop_Type; // Which Type of Stop Loss, point, price or band?
input string Stop_Style= "price";
sinput string P; // Period To Get High and Low Price When Stop_Style equals Price
input ENUM_TIMEFRAMES PP = PERIOD_M5;
sinput string StopLoss; // When Stop_Style equals Point
input int SL=50000;
sinput string TakeProfit;
input int TP=500;

sinput string Contrato; // WDO ou WIN?
input string Ativo="wdo"; // Digit Code (WDO or WIN)

sinput string BE;      // Break Even
input bool UseBreakEven=false; // Use Breakeven?
input int BreakEvenProfit=150;
input int LockProfit=0;

sinput string TS;      // Trailing Stop
input bool UseTrailingStop=false;
sinput string TrailingStop_Type; // Which Type of Trailing Stop, point, price, moving average or band?
input string Type= "moving";
input int TrailingStop=300;
input int MinimumProfit=150;

sinput string Time;
input string StartHour="09:30";
input string EndHour="17:00";
input bool CloseAtEnd=true;

sinput string Media; // Exit Rule 
input bool UseMovingAverage= true;

sinput string Maximum; // Maximum Profit/Loss 
input bool UseLimit=true;
input double MaximumProfit=5;
input double MaximumLoss=-1000;

input string MG; // Martingale Activation
input bool Martingale = true;
input string MGType= "yes"; //Agressive martingale, yes or no?
input int MagicNumber=53435;
string IndiSymbol="";
string lastPosition;
ulong last_ticket = -1;
int TradeNow;
int AccountNumber=50390953;
double last_trade_profit =0;
static double Opa2;
double Ask,Bid,bsl,btp,btp2,btp3,stp2,stp3,ssl,stp,Upper[],Lower[],middle[],partial_lots,partial_lots2,volumeCurrent,High[],Low[],HighCandle,LowCandle, stopbuy, stopsell, 
targetbuy, targetsell,VolumeLoss,fan[],Fan, Opa, divisao, divisao1, divisao2, Stop_loss_sell, Stop_loss_buy, Take_Profit_buy, Take_Profit_Sell, oz[];
datetime StartDate = D'2020.01.01 08:00';
datetime EndDate = D'2020.12.31 20:00';
bool exptime;
bool Signal = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
//--- set MagicNumber for your orders identification
   Trade.SetExpertMagicNumber(MagicNumber);

   int deviation=50;
   Trade.SetDeviationInPoints(deviation);

   IndiSymbol=Symbol();
   
   Opa2=0;     
   
// Selecting Stock
   if(Ativo == "wdo")
   {divisao = 1000;
    divisao1 = 5;
    divisao2 = 0.5;}
   if(Ativo == "win")
   {divisao = 1;
    divisao1 = 1;
    divisao2 = 5;}
/*
if(TimeCurrent()>StartDate && TimeCurrent()<EndDate){exptime=false;}else{exptime=true;}
if(exptime==true)
 {
      Alert("Hiring time expired. Please, contact the developer Vinicius B. Paiva");
      ExpertRemove();
     }
   else
     {return(0);}
*/
   if
   (AccountInfoInteger(ACCOUNT_LOGIN)!=AccountNumber)
     {
      Alert("The current Account ID:"+IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN))+" is not registered with this EA.");
      ExpertRemove();
     }
   else
     {return(0);}


//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {


  }
//+------------------------------------------------------------------+
//| Expert Ontrade function                                          |
//+------------------------------------------------------------------+
void OnTrade()
{
static long nDaysSince1970 = 0;
long n = (long)TimeCurrent() / (86400);
if ( n > nDaysSince1970 )
{
   nDaysSince1970 = n;
   Opa2 = 0;
}
        
        static int previous_open_positions = 0;
        int current_open_positions = PositionsTotal();
        if(current_open_positions < previous_open_positions)             // a position just got closed:
        {
                previous_open_positions = current_open_positions;
                HistorySelect(TimeCurrent()-300, TimeCurrent()); // 5 minutes ago
                int All_Deals = HistoryDealsTotal();
                if(All_Deals < 1) Print("Some nasty shit error has occurred :s");
                // last deal (should be an DEAL_ENTRY_OUT type):
                ulong temp_Ticket = HistoryDealGetTicket(All_Deals-1);
                string symbol=HistoryDealGetString(temp_Ticket,DEAL_SYMBOL); 
                ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(temp_Ticket,DEAL_ENTRY);
                ENUM_DEAL_TYPE entry_position=(ENUM_DEAL_TYPE)HistoryDealGetInteger(temp_Ticket,DEAL_TYPE);
                 // here check some validity factors of the position-closing deal 
                // (symbol, position ID, even MagicNumber if you care...)
                if (symbol == _Symbol)
                {
                double LAST_TRADE_PROFIT = HistoryDealGetDouble(temp_Ticket , DEAL_PROFIT);
                Print("Last Trade Profit : ", DoubleToString(LAST_TRADE_PROFIT));
                Opa = LAST_TRADE_PROFIT;
                Opa2 += Opa;
                }
                              
        }
        else if(current_open_positions > previous_open_positions)       // a position just got opened:
                previous_open_positions = current_open_positions; 

}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---     
   int period_seconds=PeriodSeconds(_Period);                     // Number of seconds in current chart period
   datetime new_time=TimeCurrent()/period_seconds*period_seconds; // Time of bar opening on current chart
   datetime new_time2=new_time+period_seconds;
   if(current_chart.isNewBar(new_time)) OnNewBar();               // When new bar appears - launch the NewBar event handler
   
// Taking Pending Orders
   ulong tickets[];
   Pending.GetTickets(_Symbol, tickets);
   int numTickets = ArraySize(tickets);
   
   if(Pending.TotalPending(_Symbol) > 1)
   {
   Delete(tickets[0]);
   }      

// Get Bollinger Bands
   double bbUpper[], bbLower[], bbMidle[];
   ArraySetAsSeries(bbUpper,true);
   ArraySetAsSeries(bbLower,true);
   ArraySetAsSeries(bbMidle,true);

   int bbHandle=iCustom(_Symbol,_Period,"Examples\\BB",BandsPeriod,BandsShift,BandsDeviation);
   CopyBuffer(bbHandle,0,0,3,bbMidle);
   CopyBuffer(bbHandle,1,0,3,bbUpper);
   CopyBuffer(bbHandle,2,0,3,bbLower);

   double bbMid= bbMidle[1];
   double bbUp = bbUpper[1];
   double bbUp2 = bbUpper[2];
   double bbLow = bbLower[1];
   double bbLow2= bbLower[2];

// New rule
   if(last_time() >= new_time && last_time() <= new_time2)
   Signal = true;
   else Signal = false; 
         
// New candle   
   if(NewCandle()){TradeNow=1;}
   Ask = SymbolInfoDouble(Symbol(),SYMBOL_ASK);
   Bid = SymbolInfoDouble(Symbol(),SYMBOL_BID);
 
// Get high 2 minutes
   ArraySetAsSeries(High,true);
   CopyHigh(_Symbol,PP,0,10,High);
   HighCandle=High[1];
  
// Get low 2 minutes

   ArraySetAsSeries(Low,true);
   CopyLow(_Symbol,PP,0,10,Low);
   LowCandle=Low[1];

   if(Stop_Style == "point")
   {
   Stop_loss_buy = iCloseTito(_Symbol,PERIOD_CURRENT,1)-SL*Point();
   Stop_loss_sell = iCloseTito(_Symbol,PERIOD_CURRENT,1)+SL*Point();
   Take_Profit_buy = Ask+TP*Point();  
   Take_Profit_Sell = Bid-TP*Point();
   }
   if(Stop_Style == "price")
   {
   Stop_loss_buy = LowCandle - (SL*Point());
   Stop_loss_sell = HighCandle + (SL*Point());
   Take_Profit_buy = HighCandle + (TP*Point());  
   Take_Profit_Sell = LowCandle - (TP*Point());
   }
   
   if(Stop_Style == "band")
   {
   long digits;
   double tickSize; 
	SymbolInfoInteger(_Symbol,SYMBOL_DIGITS,digits);
   SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE,tickSize);		
   
   Stop_loss_buy = NormalizeDouble(MathRound(bbLow/tickSize)*tickSize,digits);
   Stop_loss_sell = NormalizeDouble(MathRound(bbUp/tickSize)*tickSize,digits);
   /*
   Stop_loss_buy = LowCandle - (SL*Point());
   Stop_loss_sell = HighCandle + (SL*Point());
   */
   Take_Profit_buy = (NormalizeDouble(MathRound(bbUp/tickSize)*tickSize,digits)+TP*Point());
   Take_Profit_Sell = (NormalizeDouble(MathRound(bbLow/tickSize)*tickSize,digits)-TP*Point());
   }
   
                 
// Get High moving average 
   double ma20[];
   ArraySetAsSeries(ma20,true);

   int ma20Handle=iMA(_Symbol,MATime20,MAPeriod20,MAShift20,MAMethod20,MAPrice20);
   CopyBuffer(ma20Handle,0,0,3,ma20);
   double currentMA20=ma20[1];

// Get Low moving average 
   double ma17[];
   ArraySetAsSeries(ma17,true);

   int ma17Handle=iMA(_Symbol,MATime17,MAPeriod17,MAShift17,MAMethod17,MAPrice17);
   CopyBuffer(ma17Handle,0,0,3,ma17);
   double currentMA17=ma17[1];

// Get Activation moving average 
   double ma34[];
   ArraySetAsSeries(ma34,true);

   int ma34Handle=iMA(_Symbol,MATime34,MAPeriod34,MAShift34,MAMethod34,MAPrice34);
   CopyBuffer(ma34Handle,0,0,3,ma34);
   double currentMA34=ma34[1];

// Get 34 moving average 2 minutes
   double matwo34[];
   ArraySetAsSeries(matwo34,true);

   int ma34twoHandle=iMA(_Symbol,PERIOD_M2,MAPeriod34,MAShift34,MAMethod34,MAPrice34);
   CopyBuffer(ma34twoHandle,0,0,3,matwo34);
   double currenttwoMA34=matwo34[1];

// Get 17 moving average 2 minutes
   double matwo17[];
   ArraySetAsSeries(matwo17,true);

   int ma17twoHandle=iMA(_Symbol,PERIOD_M2,MAPeriod17,MAShift17,MAMethod17,MAPrice17);
   CopyBuffer(ma17twoHandle,0,0,3,matwo17);
   double currenttwoMA17=matwo17[1];

// open position 
   if(!WorkingHour() && CloseAtEnd)
     {CLOSEALL(0);CLOSEALL(1);
     for (int i=1; i<numTickets; i++)
     {Delete(tickets[i]);}}

   if(UseLimit==true)
   {
     double float_profit=AccountInfoDouble(ACCOUNT_PROFIT);
     if (last_profit() >= MaximumProfit || last_profit() <=MaximumLoss || (last_profit()+float_profit) >= MaximumProfit || (last_profit()+float_profit)<= MaximumLoss)
      {
         TradeNow = 0;
         Print("Profit limit was reached");
         Comment("Profit limit was reached");
            if(OpenOrders(_Symbol) > 0)
               {
                  CLOSEALL(0);CLOSEALL(1);

               }
      }

   }

// Order Rules 
      /*if((WorkingHour() && titao==true && last_position()!="Sell") && ma34[2] < ma20[2] && currentMA34 > currentMA20 && 
      iCloseTito(_Symbol,PERIOD_CURRENT, 2) < bbUp &&
      iCloseTito(_Symbol,PERIOD_CURRENT, 1) < bbUp)
     {
      if(Martingale == false)
      {OrderEntry(0);}
      if(Martingale == true)
      {
      if (MGType == "yes")
      {OrderEntry2(0);}
      if (MGType == "no")
      {OrderEntry3(0);}
      }
      }

   if((WorkingHour() && titao==true && last_position()!="Buy" ) && ma34[2] > ma17[2] && currentMA34 < currentMA17 && 
   iCloseTito(_Symbol,PERIOD_CURRENT, 2) > bbLow &&
   iCloseTito(_Symbol,PERIOD_CURRENT, 1) > bbLow )
     {
     if(Martingale == false)
     {OrderEntry(1);}
     if(Martingale == true)
     {
     if (MGType == "yes")
     {OrderEntry2(1);}
     if (MGType == "no")
     {OrderEntry3(1);}
     }
     }*/
    
      if((WorkingHour() && titao==true /*&& last_position()!="Sell"*/)  && ma20[1] > bbUp) 
     {
      if(Martingale == false)
      {OrderEntry(0);}
      if(Martingale == true)
      {
      if (MGType == "yes")
      {OrderEntry2(0);}
      if (MGType == "no")
      {OrderEntry3(0);}
      }
      }

   if((WorkingHour() && titao==true /*&& last_position()!="Buy" */) && ma17[1] < bbLow)
     {
     if(Martingale == false)
     {OrderEntry(1);}
     if(Martingale == true)
     {
     if (MGType == "yes")
     {OrderEntry2(1);}
     if (MGType == "no")
     {OrderEntry3(1);}
     }
     }
    
    if((WorkingHour() /*&& Signal == false*/ && titao_tese==true && last_position()!="Sell") && ma34[2] < ma20[2] && currentMA34 > currentMA20)
     {
      if(Martingale == false)
      {OrderEntry(0);}
      if(Martingale == true)
      {
      if (MGType == "yes")
      {OrderEntry2(0);}
      if (MGType == "no")
      {OrderEntry3(0);}
      }
      }

   if((WorkingHour() /*&& Signal == false*/&& titao_tese==true && last_position()!="Buy") && ma34[2] > ma17[2] && currentMA34 < currentMA17)
     {
     if(Martingale == false)
     {OrderEntry(1);}
     if(Martingale == true)
     {
     if (MGType == "yes")
     {OrderEntry2(1);}
     if (MGType == "no")
     {OrderEntry3(1);}
     }
     }
            
   if((WorkingHour() && nove_tres == true) && nove_ponto_tres_compra()==true && currentMA34 > currentMA20)
      {
      if(Martingale == false)
      {PendingOrderEntry(0);}
      if(Martingale == true)
      {
      if (MGType == "yes")
      {PendingOrderEntry2(0);}
      if (MGType == "no")
      {PendingOrderEntry3(0);}
      }
      }
      
   if((WorkingHour() && nove_tres == true) && nove_ponto_tres_venda()==true && currentMA34 < currentMA17)
      {
     if(Martingale == false)
     {PendingOrderEntry(1);}
     if(Martingale == true)
     {
     if (MGType == "yes")
     {PendingOrderEntry2(1);}
     if (MGType == "no")
     {PendingOrderEntry3(1);}
     }
     }
     
    if((WorkingHour() && titao_scalpe == true) && currentMA34 < currentMA20 
   && ma34[2] > ma20[2] && iCloseTito(_Symbol,PERIOD_CURRENT,1) < currentMA20 
   && iCloseTito(_Symbol,PERIOD_CURRENT,1) > currentMA17 )
     {
      if(Martingale == false)
      {//Take_Profit_buy = Ask+TP*Point();
      PendingOrderEntry(0);}
      if(Martingale == true)
      {
      if (MGType == "yes")
      {PendingOrderEntry2(0);}
      if (MGType == "no")
      {PendingOrderEntry3(0);}
      }
      }
      
   if((WorkingHour() && titao_scalpe == true) && ma34[2] < ma17[2] 
   && currentMA34 > currentMA17 && iCloseTito(_Symbol,PERIOD_CURRENT,1) > currentMA17
   && iCloseTito(_Symbol,PERIOD_CURRENT,1) < currentMA20)
    {
     if(Martingale == false)
     {//Take_Profit_Sell = Bid-TP*Point();
     PendingOrderEntry(1);}
     if(Martingale == true)
     {
     if (MGType == "yes")
     {PendingOrderEntry2(1);}
     if (MGType == "no")
     {PendingOrderEntry3(1);}
     }
     }   
     
// Exit Rule
   bool mediaSignal=false;
   if(((PositionTypeTito(_Symbol) == POSITION_TYPE_BUY) && (ma34[2] > ma17[2] && currentMA34 < currentMA17))
      || ((PositionTypeTito(_Symbol) == POSITION_TYPE_SELL) && (ma34[2] < ma20[2] && currentMA34 > currentMA20)))
     {
      mediaSignal=true;
     }

   if(UseMovingAverage==true && mediaSignal==true){CLOSEALL(0);CLOSEALL(1);}


// Tralling Stop Price
   if(UseTrailingStop==true && PositionTypeTito(_Symbol)!=-1 && Type=="price")
   {
   if(PositionTypeTito() == POSITION_TYPE_BUY)
     {TrailingStopPrice(_Symbol,LowCandle,MinimumProfit);}
   if(PositionTypeTito() == POSITION_TYPE_SELL)
     {TrailingStopPrice(_Symbol,HighCandle,MinimumProfit);}  
   }

// Tralling Stop Moving Average
   if(UseTrailingStop==true && PositionTypeTito(_Symbol)!=-1 && Type=="moving")
   {if(PositionTypeTito() == POSITION_TYPE_BUY)
     {TrailingStopPrice(_Symbol,ma20[1],MinimumProfit);}
   if(PositionTypeTito() == POSITION_TYPE_SELL)
     {TrailingStopPrice(_Symbol,ma17[1],MinimumProfit);}   
   }
   
// Tralling Stop Bolling
   if(UseTrailingStop==true && PositionTypeTito(_Symbol)!=-1 && Type=="band")
   {if(PositionTypeTito() == POSITION_TYPE_BUY)
     {TrailingStopBand(_Symbol,bbLow,MinimumProfit);}
   if(PositionTypeTito() == POSITION_TYPE_SELL)
     {TrailingStopBand(_Symbol,bbUp,MinimumProfit);}   
   }   
   
// Tralling Stop Points
     if(UseTrailingStop==true && PositionTypeTito(_Symbol)!=-1 && Type=="point")
     {TrailingStopTito(_Symbol,TrailingStop,MinimumProfit);} 
   

// Break even
   if(UseBreakEven==true && PositionTypeTito(_Symbol)!=-1)
     {
      BreakEven(_Symbol,BreakEvenProfit,LockProfit);
     }
 
   string com="";
//--- Symbol description
   string symbol=SymbolInfoString(_Symbol,SYMBOL_DESCRIPTION);
   StringAdd(com,"Symbol: "+symbol);
   StringAdd(com,"\r\n");

//--- Path to symbol
   string symbol_path=SymbolInfoString(_Symbol,SYMBOL_PATH);
   StringAdd(com,"Path: "+symbol_path);
   StringAdd(com,"\r\n");

//--- Get account currency
   string account_currency=AccountInfoString(ACCOUNT_CURRENCY);

//--- Get values of balance and equity
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   double equity=AccountInfoDouble(ACCOUNT_EQUITY);

//--- Write values of balance and equity using text formatting
   string format=StringFormat("Balance = %.2f, Equity = %.2f",balance,equity);
   StringAdd(com,format);
   StringAdd(com,"\r\n");

//--- Get values of margin and profit on account
   double margin=AccountInfoDouble(ACCOUNT_MARGIN);
   double float_profit=AccountInfoDouble(ACCOUNT_PROFIT);

//--- Write values of margin and profit using text formatting
   format=StringFormat("Margin = %.2f %%, Float Profit = %.2f ",margin,float_profit);
   StringAdd(com,format);
   StringAdd(com,"\r\n");

//--- Get values of free margin and write them using text formatting
   double free_margin=AccountInfoDouble(ACCOUNT_FREEMARGIN);
   format=StringFormat("Free Margin = %G ",free_margin);
   StringAdd(com,format);
   StringAdd(com,"\r\n");
   
//--- Add account currency name to the com string
   StringAdd(com,"Account Currency: "+account_currency);
   StringAdd(com,"\r\n");
   
//--- Get values of latest profit and write them using text formatting
   format=StringFormat("Last Trade Profit : %.2f ",last_profit_unit());
   StringAdd(com,format);
   StringAdd(com,"\r\n");   

//--- Get values of sum of profit and write them using text formatting
   format=StringFormat("Sum of Profit : %.2f ",last_profit());
   StringAdd(com,format);
   StringAdd(com,"\r\n");  

//--- Get values of sum of profit and write them using text formatting
   StringAdd(com,"Last Deal Position : "+last_position());
   StringAdd(com,"\r\n"); 

//--- Print the com string on the chart
   Comment(com);
  }
//+------------------------------------------------------------------+
//|Close price                                                       |
//+------------------------------------------------------------------+
double iCloseTito(string symbol,ENUM_TIMEFRAMES timeframe,int index)
  {
   double Close[];
   double close=0;
   ArraySetAsSeries(Close,true);
   int copied=CopyClose(symbol,timeframe,0,Bars(symbol,timeframe),Close);
   if(copied>0 && index<copied) close=Close[index];
   return(close);
  }
//+------------------------------------------------------------------+
//|open price                                               |
//+------------------------------------------------------------------+
double iOpenTito(string symbol,ENUM_TIMEFRAMES timeframe,int index)
  {
   double Open[];
   double open=0;
   ArraySetAsSeries(Open,true);
   int copied=CopyOpen(symbol,timeframe,0,Bars(symbol,timeframe),Open);
   if(copied>0 && index<copied) open=Open[index];
   return(open);
  }
//+------------------------------------------------------------------+
//|New candle                                                      |
//+------------------------------------------------------------------+
bool NewCandle()
  {
   static int BarsOnChart=0;
   if(Bars(_Symbol,PERIOD_CURRENT)==BarsOnChart)
      return (false);
   BarsOnChart=Bars(_Symbol,PERIOD_CURRENT);
   return(true);
  }  
//+------------------------------------------------------------------+
//|Last Time                                                     |
//+------------------------------------------------------------------+
datetime last_time()
  {
  int period_seconds=PeriodSeconds(_Period); 
  datetime new_time=TimeCurrent()/period_seconds*period_seconds;
  double deal_price=0;
  datetime date;
  
// --- time interval of the trade history needed
   datetime end=new_time+period_seconds;                 // current server time
   datetime start=new_time;// decrease 1 day
//--- request of trade history needed into the cache of MQL5 program
   HistorySelect(start,end);
//--- get total number of deals in the history
   int deals=HistoryDealsTotal();
//--- get ticket of the deal with the last index in the list
   ulong deal_ticket=HistoryDealGetTicket(deals-1);
   if(deal_ticket>0) // deal has been selected, let's proceed ot
     {
      //--- ticket of the order, opened the deal
      ulong order=HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
      long order_magic=HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
      long pos_ID=HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
           date = (datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME); 
           deal_price=HistoryDealGetDouble(deal_ticket,DEAL_PRICE);
           double deal_volume=HistoryDealGetDouble(deal_ticket,DEAL_VOLUME);
      PrintFormat("Deal: #%d opened by order: #%d with ORDER_MAGIC: %d was in position: #%d price: #%d volume:",
                  deals-1,order,order_magic,pos_ID,deal_price,deal_volume);

     }
   else              // error in selecting of the deal
     {
      PrintFormat("Total number of deals %d, error in selection of the deal"+
                  " with index %d. Error %d",deals,deals-1,GetLastError());
     }
   return(date);
  }
//+------------------------------------------------------------------+
//|Last Profit                                                       |
//+------------------------------------------------------------------+
datetime last_profit()
  {
// --- determine the time intervals of the required trading history
  datetime end=StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + EndHour); 
  datetime gi_time = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + StartHour);                                                                                                                     
  datetime start=end-/*gi_time;*/ PeriodSeconds(PERIOD_D1);// set the beginning time to 24 hours ago
 //--- request in the cache of the program the needed interval of the trading history
   HistorySelect(gi_time,end);
//--- obtain the number of deals in the history
   int deals=HistoryDealsTotal();

   double result=0;
   int returns=0;
   double profit=0;
   double loss=0;
//--- scan through all of the deals in the history
   for(int i=0;i<deals;i++)
     {
      //--- obtain the ticket of the deals by its index in the list
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket>0) // obtain into the cache the deal, and work with it
        
        {
         string symbol             =HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         datetime time             =(datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME);
         ulong order               =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
         long order_magic          =HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
         long pos_ID               =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
         ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);
         
          if(symbol==_Symbol)
              {
               if(entry_type==DEAL_ENTRY_OUT)
                 {
                  result+=HistoryDealGetDouble(deal_ticket,DEAL_PROFIT);
                 }
              }
         }
      }  
   return(result);
  } 
  
//+------------------------------------------------------------------+
//|Last Profit                                                       |
//+------------------------------------------------------------------+
datetime last_profit_unit()
  {
// --- determine the time intervals of the required trading history
  datetime end=StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + EndHour); 
  datetime gi_time = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + StartHour);                                                                                                                     
  datetime start=end-/*gi_time;*/ PeriodSeconds(PERIOD_D1);// set the beginning time to 24 hours ago
 //--- request in the cache of the program the needed interval of the trading history
   HistorySelect(gi_time,end);
//--- obtain the number of deals in the history
   int deals=HistoryDealsTotal();

   double result=0;
   int returns=0;
   double profit=0;
   double loss=0;
   
//--- scan through all of the deals in the history
   for(int i=0;i<deals;i++)
     {
      //--- obtain the ticket of the deals by its index in the list
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket>0) // obtain into the cache the deal, and work with it
        
        {
         string symbol             =HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         datetime time             =(datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME);
         ulong order               =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
         long order_magic          =HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
         long pos_ID               =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
         ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);
         
          if(symbol==_Symbol)
              {
               if(entry_type==DEAL_ENTRY_OUT)
                 {
                  result=HistoryDealGetDouble(deal_ticket,DEAL_PROFIT);
                 }
              }
         }
      }  
   return(result);
  }   
//+------------------------------------------------------------------+
//|Function Last Sell or Buy                                         |
//+------------------------------------------------------------------+
string last_position()
  {
// --- determine the time intervals of the required trading history
  datetime end=StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + EndHour); 
  datetime gi_time = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + StartHour);                                                                                                                     
  datetime start=end-/*gi_time;*/ PeriodSeconds(PERIOD_D1);// set the beginning time to 24 hours ago
 //--- request in the cache of the program the needed interval of the trading history
   HistorySelect(gi_time,end);
//--- obtain the number of deals in the history
   int deals=HistoryDealsTotal();
   lastPosition="No Orders";
//--- scan through all of the deals in the history
   for(int i=0;i<deals;i++)
     {
      //--- obtain the ticket of the deals by its index in the list
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket>0) // obtain into the cache the deal, and work with it
        
        {
         string symbol             =HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         datetime time             =(datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME);
         ulong order               =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
         long order_magic          =HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
         long pos_ID               =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
         ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);
         ENUM_DEAL_TYPE entry_position=(ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket,DEAL_TYPE);
         
          if(symbol==_Symbol)
              {
               if(entry_type==DEAL_ENTRY_OUT)
                 {
                  if(entry_position == DEAL_TYPE_BUY)
                     {
                     lastPosition="Buy";
                     }
                  else if(entry_position == DEAL_TYPE_SELL)
                     {
                     lastPosition="Sell";
                     }
                  else
                     {
                     lastPosition="No Orders";
                     }          
                 }
              }
         }
      }  
     return(lastPosition);
  } 
//+------------------------------------------------------------------+
//|Working Hour                                                        |
//+------------------------------------------------------------------+
bool WorkingHour()
  {
   datetime gi_time_01 = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + StartHour);
   datetime gi_time_02 = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + EndHour);
   datetime datetime_0 = TimeCurrent();

   if( gi_time_01 < gi_time_02 && gi_time_01 <= datetime_0 && datetime_0 <= gi_time_02 ) return (true);
   if( gi_time_01 > gi_time_02 && (datetime_0 >= gi_time_01 || datetime_0 <= gi_time_02) ) return (true);


   return (false);
  }
//+------------------------------------------------------------------+
//|Close all                                                      |
//+------------------------------------------------------------------+
void CLOSEALL(int type)
  {
   int _tp=PositionsTotal();
   for(int i=_tp-1; i>=0; i--)
     {
      string _p_symbol=PositionGetSymbol(i);
      ulong tick=PositionGetTicket(i);

      if(PositionSelectByTicket(tick))
        {

         if(MagicNumber==PositionGetInteger(POSITION_MAGIC))
           {
            if(type==0 && PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
              {Trade.PositionClose(tick,-1);}
            if(type==1 && PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
              {

               Trade.PositionClose(tick,-1);
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//|Open order                                                        |
//+------------------------------------------------------------------+
int OpenOrders(string symbol)
  {
   int tnum=0;
   int _tp=PositionsTotal();

   for(int i=_tp-1; i>=0; i--)
     {
      string _p_symbol=PositionGetSymbol(i);
      ulong tick=PositionGetTicket(i);

      if(PositionSelectByTicket(tick))
        {
         if(_p_symbol==symbol)
           {
            if(MagicNumber==PositionGetInteger(POSITION_MAGIC))
              {
               tnum++;
              }
           }
        }
     }
   
   return(tnum);
  }
//+------------------------------------------------------------------+
//|Entry                                                             |
//+------------------------------------------------------------------+
void OrderEntry(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);

   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
     {

      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");

      TradeNow=0;
     }

   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
     {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");

      TradeNow=0;
     }
  }
//+------------------------------------------------------------------+
//|Pending Entry                                                             |
//+------------------------------------------------------------------+
void PendingOrderEntry(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);
int period_seconds=PeriodSeconds(_Period);                     // Number of seconds in current chart period
   datetime new_time=TimeCurrent()/period_seconds*period_seconds; // Time of bar opening on current chart
   datetime expiration=new_time+period_seconds;
   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
     {

      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
     }

   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
     {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
     }
  }
//+------------------------------------------------------------------+
//|Pending Entry                                                            |
//+------------------------------------------------------------------+
void PendingOrderEntry2(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);
int period_seconds=PeriodSeconds(_Period);                     // Number of seconds in current chart period
   datetime new_time=TimeCurrent()/period_seconds*period_seconds; // Time of bar opening on current chart
   datetime expiration=new_time+period_seconds;
   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if(last_profit() >= 0 && last_profit_unit() >= 0)
      {
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
     
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(newvolume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(newvolume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())     
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(newvolume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
       if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(newvolume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
     
        if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(newvolume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if (last_profit() >= 0 && last_profit_unit() >= 0)
      {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
   
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
    
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(newvolume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade"); //newvolume

      TradeNow=0;
      
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      Comment(newvolume);
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(newvolume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
      if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
     
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(newvolume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
     
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(newvolume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
    
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(newvolume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
 
  }      
//+------------------------------------------------------------------+
//|Pending Entry                                                     |
//+------------------------------------------------------------------+
void PendingOrderEntry3(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);
int period_seconds=PeriodSeconds(_Period);                     // Number of seconds in current chart period
   datetime new_time=TimeCurrent()/period_seconds*period_seconds; // Time of bar opening on current chart
   datetime expiration=new_time+period_seconds;
   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if(last_profit() >= 0 && last_profit_unit() >= 0)
      {
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Take_Profit_buy;}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
     
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=HighCandle+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=HighCandle+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())     
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=HighCandle+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
       if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=HighCandle+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
     
        if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=HighCandle+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.BuyStop(Volume,HighCandle,NULL,bsl,btp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if (last_profit() >= 0 && last_profit_unit() >= 0)
      {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Take_Profit_Sell;}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");

      TradeNow=0;
   
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
         
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=LowCandle-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade"); //newvolume

      TradeNow=0;
      
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=LowCandle-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
      if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
          
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=LowCandle-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=LowCandle-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
          
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=LowCandle-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.SellStop(Volume,LowCandle,NULL,ssl,stp,ORDER_TIME_GTC,Time[0]+Period()*60,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
 
  } 
//+------------------------------------------------------------------+
//|Position Type                                                     |
//+------------------------------------------------------------------+
long PositionTypeTito(string pSymbol=NULL)
  {
   if(pSymbol==NULL) pSymbol=_Symbol;
   bool select=PositionSelect(pSymbol);
   if(select == true) return(PositionGetInteger(POSITION_TYPE));
   else return(WRONG_VALUE);
  }
//+------------------------------------------------------------------+
//|Tralling Stop                                                     |
//+------------------------------------------------------------------+
bool TrailingStopTito(string pSymbol,int pTrailPoints,int pMinProfit=0)
  {
   if(PositionSelect(pSymbol)==true && pTrailPoints>0)
     {
      enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      request.action = TRADE_ACTION_SLTP;
      request.symbol = pSymbol;

      long posType=PositionGetInteger(POSITION_TYPE);
      double currentStop=PositionGetDouble(POSITION_SL);
      double openPrice=PositionGetDouble(POSITION_PRICE_OPEN);

      double point=SymbolInfoDouble(pSymbol,SYMBOL_POINT);
      int digits=(int)SymbolInfoInteger(pSymbol,SYMBOL_DIGITS);
      
      double minProfit = pMinProfit * point;
      double trailStop = pTrailPoints * point;
      currentStop=NormalizeDouble(currentStop,digits);

      double trailStopPrice;
      double currentProfit;
      btp=openPrice+TP*Point();
      btp=NormalizeDouble(btp,Digits());
      stp=openPrice-TP*Point();
      stp=NormalizeDouble(stp,Digits());
      // Order loop
      int retryCount=0;
      int checkRes=0;

      do
        {
         if(posType==POSITION_TYPE_BUY)
           {
            trailStopPrice = SymbolInfoDouble(pSymbol,SYMBOL_BID) - trailStop;
            trailStopPrice = NormalizeDouble(trailStopPrice,digits);
            currentProfit=SymbolInfoDouble(pSymbol,SYMBOL_BID)-openPrice;

            if(trailStopPrice>currentStop && currentProfit>=minProfit)
              {
               request.sl= trailStopPrice;
               request.tp = btp;
               bool sent = OrderSend(request,result);
              }
            else return(false);
           }
         else if(posType==POSITION_TYPE_SELL)
           {
            trailStopPrice = SymbolInfoDouble(pSymbol,SYMBOL_ASK) + trailStop;
            trailStopPrice = NormalizeDouble(trailStopPrice,digits);
            currentProfit=openPrice-SymbolInfoDouble(pSymbol,SYMBOL_ASK);

            if((trailStopPrice<currentStop || currentStop==0) && currentProfit>=minProfit)
              {
               request.sl= trailStopPrice;
               request.tp = stp;
               bool sent = OrderSend(request,result);
              }
            else return(false);
           }

         checkRes=CheckReturnCode(result.retcode);

         if(checkRes==CHECK_RETCODE_OK) break;
         else if(checkRes==CHECK_RETCODE_ERROR)
           {
            string errDesc=TradeServerReturnCodeDescription(result.retcode);
            Alert("Trailing stop: Error ",result.retcode," - ",errDesc);
            break;
           }
         else
           {
            Print("Server error detected, retrying...");
            Sleep(RETRY_DELAY);
            retryCount++;
           }
        }
      while(retryCount<MAX_RETRIES);

      if(retryCount>=MAX_RETRIES)
        {
         string errDesc=TradeServerReturnCodeDescription(result.retcode);
         Alert("Max retries exceeded: Error ",result.retcode," - ",errDesc);
        }

      string errDesc=TradeServerReturnCodeDescription(result.retcode);
      Print("Trailing stop: ",result.retcode," - ",errDesc,", Old SL: ",currentStop,", New SL: ",request.sl,", Bid: ",SymbolInfoDouble(pSymbol,SYMBOL_BID),", Ask: ",SymbolInfoDouble(pSymbol,SYMBOL_ASK),", Stop Level: ",SymbolInfoInteger(pSymbol,SYMBOL_TRADE_STOPS_LEVEL));

      if(checkRes == CHECK_RETCODE_OK) return(true);
      else return(false);
     }

   else return(false);
  }  
//+------------------------------------------------------------------+
//|Check Return                                                      |
//+------------------------------------------------------------------+
int CheckReturnCode(uint pRetCode)
  {
    enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
   int status;
   switch(pRetCode)
     {
      
      case TRADE_RETCODE_REQUOTE:
      case TRADE_RETCODE_CONNECTION:
      case TRADE_RETCODE_PRICE_CHANGED:
      case TRADE_RETCODE_TIMEOUT:
      case TRADE_RETCODE_PRICE_OFF:
      case TRADE_RETCODE_REJECT:
      case TRADE_RETCODE_ERROR:

         status=CHECK_RETCODE_RETRY;
         break;

      case TRADE_RETCODE_DONE:
      case TRADE_RETCODE_DONE_PARTIAL:
      case TRADE_RETCODE_PLACED:
      case TRADE_RETCODE_NO_CHANGES:

         status=CHECK_RETCODE_OK;
         break;

      default: status=CHECK_RETCODE_ERROR;
     }

   return(status);
  }
//+------------------------------------------------------------------+
//|Break Even                                                        |
//+------------------------------------------------------------------+
bool BreakEven(string pSymbol,int pBreakEven,int pLockProfit=0)
  {
   if(PositionSelect(pSymbol)==true && pBreakEven>0)
     {
       enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      request.action = TRADE_ACTION_SLTP;
      request.symbol = pSymbol;

      long posType=PositionGetInteger(POSITION_TYPE);
      double currentSL = PositionGetDouble(POSITION_SL);
      double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);

      double point=SymbolInfoDouble(pSymbol,SYMBOL_POINT);
      int digits=(int)SymbolInfoInteger(pSymbol,SYMBOL_DIGITS);

      double breakEvenStop;
      double currentProfit;
      btp=openPrice+TP*Point();
      btp=NormalizeDouble(btp,Digits());
      stp=openPrice-TP*Point();
      stp=NormalizeDouble(stp,Digits());

      int retryCount=0;
      int checkRes=0;

      double bid=0,ask=0;

      do
        {
         if(posType==POSITION_TYPE_BUY)
           {
            bid=SymbolInfoDouble(pSymbol,SYMBOL_BID);
            breakEvenStop = openPrice + (pLockProfit * point);
            currentProfit = bid - openPrice;

            breakEvenStop = NormalizeDouble(breakEvenStop, digits);
            currentProfit = NormalizeDouble(currentProfit, digits);

            if(currentSL<breakEvenStop && currentProfit>=pBreakEven*point)
              {
               request.sl = breakEvenStop;
               request.tp = btp;
               bool sent=OrderSend(request,result);
              }
            else return(false);
           }
         else if(posType==POSITION_TYPE_SELL)
           {
            ask=SymbolInfoDouble(pSymbol,SYMBOL_ASK);
            breakEvenStop = openPrice - (pLockProfit * point);
            currentProfit = openPrice - ask;

            breakEvenStop = NormalizeDouble(breakEvenStop, digits);
            currentProfit = NormalizeDouble(currentProfit, digits);

            if((currentSL>breakEvenStop || currentSL==0) && currentProfit>=pBreakEven*point)
              {
               request.sl = breakEvenStop;
               request.tp = stp;
               bool sent=OrderSend(request,result);
              }
            else return(false);
           }

         checkRes=CheckReturnCode(result.retcode);

         if(checkRes==CHECK_RETCODE_OK) break;
         else if(checkRes==CHECK_RETCODE_ERROR)
           {
            string errDesc=TradeServerReturnCodeDescription(result.retcode);
            Alert("Break even stop: Error ",result.retcode," - ",errDesc);
            break;
           }
         else
           {
            Print("Server error detected, retrying...");
            Sleep(RETRY_DELAY);
            retryCount++;
           }
        }
      while(retryCount<MAX_RETRIES);

      if(retryCount>=MAX_RETRIES)
        {
         string errDesc=TradeServerReturnCodeDescription(result.retcode);
         Alert("Max retries exceeded: Error ",result.retcode," - ",errDesc);
        }

      string errDesc=TradeServerReturnCodeDescription(result.retcode);
      Print("Break even stop: ",result.retcode," - ",errDesc,", SL: ",request.sl,", Bid: ",bid,", Ask: ",ask,", Stop Level: ",SymbolInfoInteger(pSymbol,SYMBOL_TRADE_STOPS_LEVEL));

      if(checkRes == CHECK_RETCODE_OK) return(true);
      else return(false);
     }
   else return(false);
  }
//+------------------------------------------------------------------+
//|Ensure Function                                                   |
//+------------------------------------------------------------------+
void func_EnsureLotWithinAllowedLimits(double &chng_Lot,string _symbol)
  {
//get minimum, maximum and step-size permitted for a lot
   double lcl_MinPermittedLot,lcl_MaxPermittedLot,lcl_MinPermittedLotStep;
   lcl_MinPermittedLot = MarketInfo( _symbol, MODE_MINLOT );
   lcl_MaxPermittedLot = MarketInfo( _symbol, MODE_MAXLOT );
   lcl_MinPermittedLotStep=MarketInfo(_symbol,MODE_LOTSTEP);
   int _LotDigits=4;
   double micro_lot=0.01,mini_lot=0.1,lot1=1;
//Print("Lots1: ",chng_Lot);
   if(NormalizeDouble(chng_Lot,_LotDigits)!=NormalizeDouble(MathRound(chng_Lot/lcl_MinPermittedLotStep)*lcl_MinPermittedLotStep,_LotDigits))
     {
      //ensure given lot follows lot-step
      chng_Lot=MathRound(chng_Lot/lcl_MinPermittedLotStep)*lcl_MinPermittedLotStep;
      //Print("Lots2: ",chng_Lot);
     }
   if((NormalizeDouble(chng_Lot,_LotDigits)<NormalizeDouble(lcl_MinPermittedLot,_LotDigits)))
     {
      // lot must not be below the minimum allowed limit
      chng_Lot=lcl_MinPermittedLot;
     }
   else if(NormalizeDouble(chng_Lot,_LotDigits)>NormalizeDouble(lcl_MaxPermittedLot,_LotDigits))
     {
      // lot must not be above the maximum allowed limit
      chng_Lot=lcl_MaxPermittedLot;
     }
// normalize the lot
   double _LotMicro=0.01,// micro lots
   _LotMini=0.10,// mini lots
   _LotNrml= 1.00;
//
   if(lcl_MinPermittedLot==_LotMicro)
      _LotDigits=2;
   else if(lcl_MinPermittedLot==_LotMini)
      _LotDigits=1;
   else if(lcl_MinPermittedLot==_LotNrml)
      _LotDigits=0;
   chng_Lot=NormalizeDouble(chng_Lot,_LotDigits);
  }
//+------------------------------------------------------------------+
//| New bar event handler function                                   |
//+------------------------------------------------------------------+
void OnNewBar()
  {
   PrintFormat("New bar: %s",TimeToString(TimeCurrent(),TIME_SECONDS));
  }    
//+------------------------------------------------------------------+
//|Tralling Stop Price                                               |
//+------------------------------------------------------------------+
bool TrailingStopPrice(string pSymbol,double pTrailPrice,int pMinProfit=0)
  {
  if(PositionSelect(pSymbol) == true && pTrailPrice > 0)
	{
	    enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
	   MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
		request.action = TRADE_ACTION_SLTP;
		request.symbol = pSymbol;
		
		long posType = PositionGetInteger(POSITION_TYPE);
		double currentStop = PositionGetDouble(POSITION_SL);
		double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
		double point = SymbolInfoDouble(pSymbol,SYMBOL_POINT);
		
		long digits;
		double tickSize; 
		SymbolInfoInteger(pSymbol,SYMBOL_DIGITS,digits);
      SymbolInfoDouble(pSymbol,SYMBOL_TRADE_TICK_SIZE,tickSize);
		
		/*
		int digits = (int)SymbolInfoInteger(pSymbol,SYMBOL_DIGITS);
		*/
	
		double minProfit = pMinProfit * point;
				
		currentStop=NormalizeDouble(MathRound(currentStop/tickSize)*tickSize,digits);
		pTrailPrice=NormalizeDouble(MathRound(pTrailPrice/tickSize)*tickSize,digits);
		
		/*
		currentStop = NormalizeDouble(currentStop,digits);
		pTrailPrice = NormalizeDouble(pTrailPrice,digits);
		*/
		
		double currentProfit;
		btp=openPrice+TP*Point();
      btp=NormalizeDouble(btp,Digits());
      stp=openPrice-TP*Point();
      stp=NormalizeDouble(stp,Digits());
		
		int retryCount = 0;
		int checkRes = 0;
		
		double bid = 0, ask = 0;
		
		do 
		{
			if(posType == POSITION_TYPE_BUY)
			{
				bid = SymbolInfoDouble(pSymbol,SYMBOL_BID);
				currentProfit = bid - openPrice;
				if(pTrailPrice > currentStop && currentProfit >= minProfit) 
				{
					request.sl = pTrailPrice;
					request.tp = btp;
					bool sent = OrderSend(request,result);
				}
				else return(false);
			}
			else if(posType == POSITION_TYPE_SELL)
			{
				ask = SymbolInfoDouble(pSymbol,SYMBOL_ASK);
				currentProfit = openPrice - ask;
				if((pTrailPrice < currentStop  || currentStop == 0) && currentProfit >= minProfit)
				{
					request.sl = pTrailPrice;
					request.tp = stp;
					bool sent = OrderSend(request,result);
				}
				else return(false);
			}
			
			checkRes = CheckReturnCode(result.retcode);
		
			if(checkRes == CHECK_RETCODE_OK) break;
			else if(checkRes == CHECK_RETCODE_ERROR)
			{
				string errDesc = TradeServerReturnCodeDescription(result.retcode);
				Alert("Trailing stop: Error ",result.retcode," - ",errDesc);
				break;
			}
			else
			{
				Print("Server error detected, retrying...");
				Sleep(RETRY_DELAY);
				retryCount++;
			}
		}
		while(retryCount < MAX_RETRIES);
	
		if(retryCount >= MAX_RETRIES)
		{
			string errDesc = TradeServerReturnCodeDescription(result.retcode);
			Alert("Max retries exceeded: Error ",result.retcode," - ",errDesc);
		}
		
		string errDesc = TradeServerReturnCodeDescription(result.retcode);
		Print("Trailing stop: ",result.retcode," - ",errDesc,", Old SL: ",currentStop,", New SL: ",request.sl,", Bid: ",bid,", Ask: ",ask,", Stop Level: ",SymbolInfoInteger(pSymbol,SYMBOL_TRADE_STOPS_LEVEL));
		
		if(checkRes == CHECK_RETCODE_OK) return(true);
		else return(false);
	}
	else return(false);
} 
//+------------------------------------------------------------------+
//|Function Calcprofit                                               |
//+------------------------------------------------------------------+
bool CalcProfit()
  {
// --- determine the time intervals of the required trading history
   datetime end=StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + EndHour); 
   datetime gi_time = StringToTime(TimeToString(TimeCurrent(), TIME_DATE) + " " + StartHour);                                                                                                                     
   datetime start=end-/*gi_time;*/ PeriodSeconds(PERIOD_D1);// set the beginning time to 24 hours ago

//--- request in the cache of the program the needed interval of the trading history
   HistorySelect(gi_time,end);
//--- obtain the number of deals in the history
   int deals=HistoryDealsTotal();

   double result=0;
   int returns=0;
   double profit=0;
   double loss=0;
//--- scan through all of the deals in the history
   for(int i=0;i<deals;i++)
     {
      //--- obtain the ticket of the deals by its index in the list
      ulong deal_ticket=HistoryDealGetTicket(i);
      if(deal_ticket>0) // obtain into the cache the deal, and work with it
        
        {
         string symbol             =HistoryDealGetString(deal_ticket,DEAL_SYMBOL);
         datetime time             =(datetime)HistoryDealGetInteger(deal_ticket,DEAL_TIME);
         ulong order               =HistoryDealGetInteger(deal_ticket,DEAL_ORDER);
         long order_magic          =HistoryDealGetInteger(deal_ticket,DEAL_MAGIC);
         long pos_ID               =HistoryDealGetInteger(deal_ticket,DEAL_POSITION_ID);
         ENUM_DEAL_ENTRY entry_type=(ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket,DEAL_ENTRY);
         //--- proceed deal with specified DEAL_MAGIC
         if(order_magic==MagicNumber)
           {
            //... processing of deal with some DEAL_MAGIC
           
            if(symbol==_Symbol)
              {
               if(entry_type==DEAL_ENTRY_OUT)
                 returns++;
                 {
                  result+=HistoryDealGetDouble(deal_ticket,DEAL_PROFIT);
                   if(result>0)
                    {
                    profit=result;
                    }
                  if(result<0)
                    {
                    loss=result;
                    }
                 
                  if((result>=MaximumProfit) || (result+Position.Profit()>= MaximumProfit))
                    {
                   return(true);
                    }
                  if((result<=MaximumLoss) ||  (result+Position.Profit()<=MaximumLoss))
                    {                                     
                    return(true);
                    }
                 }
              }
           }

        }
      else // unsuccessful attempt to obtain a deal
        {
         PrintFormat("We couldn't select a deal, with the index %d. Error %d",
                     i,GetLastError());
        }
     }
//--- output the results of the calculations
   PrintFormat("The total number of %d deals with a financial result. Profit=%.2f , Loss= %.2f",
               returns,profit,loss,result);       
              
               
   return(false);
  } 
//+------------------------------------------------------------------+
//|Setup 9.3 Compra                                                  |
//+------------------------------------------------------------------+  
 bool nove_ponto_tres_compra()
  {
double matwo9[];
ArraySetAsSeries(matwo9,true);

int ma9twoHandle=iMA(_Symbol,PERIOD_CURRENT,9,MAShift34,MAMethod34,MAPrice34);
CopyBuffer(ma9twoHandle,0,0,3,matwo9);
double currenttwoMA9=matwo9[1];
   
    
if(LowCandle <= currenttwoMA9 && iOpenTito(_Symbol,PERIOD_CURRENT,1) >= currenttwoMA9 && 
iCloseTito(_Symbol,PERIOD_CURRENT,1) > currenttwoMA9 && currenttwoMA9 > matwo9[2])
  {
  return(true);
  }
  return(false);
  }  
//+------------------------------------------------------------------+
//|Setup 9.3 Venda                                                   |
//+------------------------------------------------------------------+  
 bool nove_ponto_tres_venda()
  {
double matwo9[];
ArraySetAsSeries(matwo9,true);

int ma9twoHandle=iMA(_Symbol,PERIOD_CURRENT,9,MAShift34,MAMethod34,MAPrice34);
CopyBuffer(ma9twoHandle,0,0,3,matwo9);
double currenttwoMA9=matwo9[1];
     
if(HighCandle >= currenttwoMA9 && iOpenTito(_Symbol,PERIOD_CURRENT,1) <= currenttwoMA9 && 
iCloseTito(_Symbol,PERIOD_CURRENT,1) < currenttwoMA9 && currenttwoMA9 < matwo9[2])
  {
  return(true);
  }
  return(false);
  } 
//+------------------------------------------------------------------+
//|Log Trade Request                                                 |
//+------------------------------------------------------------------+   
void LogTradeRequest()
{
   MqlTradeRequest request;
   MqlTradeResult result;
   Print("MqlTradeRequest - action:",request.action,", comment:",request.comment,", deviation:",request.deviation,", expiration:",request.expiration,", magic:",request.magic,", order:",request.order,", position:",request.position,", position_by:",request.position_by,", price:",request.price,", ls:",request.sl,", stoplimit:",request.stoplimit,", symbol:",request.symbol,", tp:",request.tp,", type:",request.type,", type_filling:",request.type_filling,", type_time:",request.type_time,", volume:",request.volume);
   Print("MqlTradeResult - ask:",result.ask,", bid:",result.bid,", comment:",result.comment,", deal:",result.deal,", order:",result.order,", price:",result.price,", request_id:",result.request_id,", retcode:",result.retcode,", retcode_external:",result.retcode_external,", volume:",result.volume);
}   
//+------------------------------------------------------------------+
//|Delete Pending Order                                              |
//+------------------------------------------------------------------+ 
bool Delete(ulong pTicket)
{
	 enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
	MqlTradeRequest request;
   MqlTradeResult result;
	ZeroMemory(request);
	ZeroMemory(result);
	
	request.action = TRADE_ACTION_REMOVE;
	request.order = pTicket;
	
	// Order loop
	int retryCount = 0;
	int checkCode = 0;
	
	do 
	{
		bool sent = OrderSend(request,result);
		
		checkCode = CheckReturnCode(result.retcode);
		
		if(checkCode == CHECK_RETCODE_OK) break;
		else if(checkCode == CHECK_RETCODE_ERROR)
		{
			string errDesc = TradeServerReturnCodeDescription(result.retcode);
			Alert("Delete order: Error ",result.retcode," - ",errDesc);
			LogTradeRequest();
			break;
		}
		else
		{
			Print("Server error detected, retrying...");
			Sleep(RETRY_DELAY);
			retryCount++;
		}
	}
	while(retryCount < MAX_RETRIES);
	
	if(retryCount >= MAX_RETRIES)
	{
		string errDesc = TradeServerReturnCodeDescription(result.retcode);
		Alert("Max retries exceeded: Error ",result.retcode," - ",errDesc);
	}
	
	string errDesc = TradeServerReturnCodeDescription(result.retcode);
	Print("Delete order #",pTicket,": ",result.retcode," - ",errDesc);
	
	if(checkCode == CHECK_RETCODE_OK) 
	{
		Comment("Pending order ",pTicket," deleted");
		return(true);
	}
	else return(false);
}

//+------------------------------------------------------------------+
//|Entry                                                             |
//+------------------------------------------------------------------+
void OrderEntry2(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);

   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if(last_profit() >= 0 && last_profit_unit() >= 0)
      {
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");

      TradeNow=0;
     
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(newvolume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(newvolume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())     
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(newvolume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
       if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(newvolume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
     
        if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(newvolume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if (last_profit() >= 0 && last_profit_unit() >= 0)
      {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");

      TradeNow=0;
   
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
    
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(newvolume,NULL,Bid,ssl,stp,"EA Trade"); //newvolume

      TradeNow=0;
      
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
      Comment(newvolume);
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(newvolume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
      if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
     
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(newvolume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
     
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(newvolume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newvolume = (newprofit/divisao1*divisao2)/(TP/divisao);
      newvolume = MathCeil(newvolume);
    
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(newvolume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
 
  }
  
//+------------------------------------------------------------------+
//|Entry                                                             |
//+------------------------------------------------------------------+
void OrderEntry3(int direction)
  {
//double lotsize=NormalizeDouble(LotSize/4,2);

   if(direction==0 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if(last_profit() >= 0 && last_profit_unit() >= 0)
      {
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(TP>0)
        {btp=Ask+TP*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");

      TradeNow=0;
     
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=Ask+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)      
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=Ask+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())     
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=Ask+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
       if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=Ask+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
     
        if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())     
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit ;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0)
        {bsl=Stop_loss_buy;}
      else
        {bsl=0;}
      if(newtp>0)
        {btp=Ask+newtp*Point();}else{btp=0;}

      btp=NormalizeDouble(btp,Digits());
      ssl=NormalizeDouble(bsl,Digits());
      Trade.Buy(Volume,NULL,Ask,bsl,btp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
   if(direction==1 && TradeNow==1 && OpenOrders(Symbol())<1)
      {
      
      if (last_profit() >= 0 && last_profit_unit() >= 0)
      {

      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(TP>0){stp=Bid-TP*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");

      TradeNow=0;
   
     }
     if (last_profit() >= 0 && last_profit_unit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
         
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=Bid-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade"); //newvolume

      TradeNow=0;
      
     }
          if (last_profit_unit() >= 0 && last_profit() < 0)
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
            
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=Bid-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
      if (last_profit() < 0 && last_profit_unit() < 0 && last_profit_unit()<last_profit())
     {
      double newprofit = (MaximumProfit - (last_profit_unit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
          
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=Bid-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
    
     }
          if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()<last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
           
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=Bid-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
      
     }
     if (last_profit() < 0 && last_profit_unit() < 0 && last_profit()==last_profit_unit())
     {
      double newprofit = (MaximumProfit - (last_profit())) + MaximumProfit;
      double newtp = (((newprofit*divisao2)/divisao1)*divisao)/Volume;
          
      if(SL>0){ssl=Stop_loss_sell;}else{ssl=0;}
      if(newtp>0){stp=Bid-newtp*Point();}else{stp=0;}

      stp=NormalizeDouble(stp,Digits());
      ssl=NormalizeDouble(ssl,Digits());
      Trade.Sell(Volume,NULL,Bid,ssl,stp,"EA Trade");//newvolume

      TradeNow=0;
     
     }
     }
 
  } 
  
//+------------------------------------------------------------------+
//|Tralling Stop Band                                             |
//+------------------------------------------------------------------+
bool TrailingStopBand(string pSymbol,double pTrailPrice,int pMinProfit=0)
  {
  if(PositionSelect(pSymbol) == true && pTrailPrice > 0)
	{
	    enum ENUM_CHECK_RETCODE
        {
         CHECK_RETCODE_OK,
         CHECK_RETCODE_ERROR,
         CHECK_RETCODE_RETRY
        };
	   MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
		request.action = TRADE_ACTION_SLTP;
		request.symbol = pSymbol;
		
		long posType = PositionGetInteger(POSITION_TYPE);
		double currentStop = PositionGetDouble(POSITION_SL);
		double currentTStop = PositionGetDouble(POSITION_TP);
		double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
		double point = SymbolInfoDouble(pSymbol,SYMBOL_POINT);
		
		long digits;
		double tickSize; 
		SymbolInfoInteger(pSymbol,SYMBOL_DIGITS,digits);
      SymbolInfoDouble(pSymbol,SYMBOL_TRADE_TICK_SIZE,tickSize);
		
		/*
		int digits = (int)SymbolInfoInteger(pSymbol,SYMBOL_DIGITS);
		*/
	
		double minProfit = pMinProfit * point;
				
		currentStop=NormalizeDouble(MathRound(currentStop/tickSize)*tickSize,digits);
		pTrailPrice=NormalizeDouble(MathRound(pTrailPrice/tickSize)*tickSize,digits);
		
		/*
		currentStop = NormalizeDouble(currentStop,digits);
		pTrailPrice = NormalizeDouble(pTrailPrice,digits);
		*/
		
		double currentProfit;
		btp=openPrice+TP*Point();
      btp=NormalizeDouble(btp,Digits());
      stp=openPrice-TP*Point();
      stp=NormalizeDouble(stp,Digits());
		
		int retryCount = 0;
		int checkRes = 0;
		
		double bid = 0, ask = 0;
		
		do 
		{
			if(posType == POSITION_TYPE_BUY)
			{
				bid = SymbolInfoDouble(pSymbol,SYMBOL_BID);
				currentProfit = bid - openPrice;
				if(pTrailPrice > currentStop && currentProfit >= minProfit) 
				{
					request.sl = pTrailPrice;
					request.tp = currentTStop;
					bool sent = OrderSend(request,result);
				}
				else return(false);
			}
			else if(posType == POSITION_TYPE_SELL)
			{
				ask = SymbolInfoDouble(pSymbol,SYMBOL_ASK);
				currentProfit = openPrice - ask;
				if((pTrailPrice < currentStop  || currentStop == 0) && currentProfit >= minProfit)
				{
					request.sl = pTrailPrice;
					request.tp = currentTStop;
					bool sent = OrderSend(request,result);
				}
				else return(false);
			}
			
			checkRes = CheckReturnCode(result.retcode);
		
			if(checkRes == CHECK_RETCODE_OK) break;
			else if(checkRes == CHECK_RETCODE_ERROR)
			{
				string errDesc = TradeServerReturnCodeDescription(result.retcode);
				Alert("Trailing stop: Error ",result.retcode," - ",errDesc);
				break;
			}
			else
			{
				Print("Server error detected, retrying...");
				Sleep(RETRY_DELAY);
				retryCount++;
			}
		}
		while(retryCount < MAX_RETRIES);
	
		if(retryCount >= MAX_RETRIES)
		{
			string errDesc = TradeServerReturnCodeDescription(result.retcode);
			Alert("Max retries exceeded: Error ",result.retcode," - ",errDesc);
		}
		
		string errDesc = TradeServerReturnCodeDescription(result.retcode);
		Print("Trailing stop: ",result.retcode," - ",errDesc,", Old SL: ",currentStop,", New SL: ",request.sl,", Bid: ",bid,", Ask: ",ask,", Stop Level: ",SymbolInfoInteger(pSymbol,SYMBOL_TRADE_STOPS_LEVEL));
		
		if(checkRes == CHECK_RETCODE_OK) return(true);
		else return(false);
	}
	else return(false);
}    
//+------------------------------------------------------------------+
