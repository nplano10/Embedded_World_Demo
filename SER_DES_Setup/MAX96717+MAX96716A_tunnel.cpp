/*
# Name: kwang3
# Date: 2/4/2025
# Version: 6.5.0
#
# THIS DATA FILE, AND ALL INFORMATION CONTAINED THEREIN,
# IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATA FILE,
# THE INFORMATION CONTAINED THEREIN, OR ITS USE FOR ANY PURPOSE.
# BEFORE USING THIS DATA FILE IN ANY APPLICATION FOR PRODUCTION OR DEPLOYMENT,
# THE CUSTOMER IS SOLELY RESPONSIBLE FOR TESTING AND VERIFYING
# THE CONTENT OF THIS DATA FILE IN CONNECTION WITH THEIR PRODUCTS AND SYSTEM(S).
# ---------------------------------------------------------------------------------
#  ____      ____  
# |  _ \    / _  | 
# | | \ \  / / | | 
# | |  \ \/ /  | | 
# | |   \/ /   | | 
# | |   / /\   | | 
# | |  / /\ \  | | 
# | | / /  \ \ | | 
# |_|/_/    \_\|_| 
#
# ---------------------------------------------------------------------------------
*/
/*
# This script is validated on: 
# MAX96717
# MAX96716A
# Please refer to the Errata sheet for each device.
# ---------------------------------------------------------------------------------
*/
// GMSL-A / Serializer: MAX96717 (Tunnel Mode) / Mode: 1x4 / Device Address: 0x80 / Multiple-VC Case: Single VC / Multiple-VC Pipe Sharing: N/A
// PipeZ:
// Input Stream: VC0 YUV422_10bit PortB (D-PHY)

// Deserializer: MAX96716A / Mode: 2 (1x4) / Device Address: 0x98
// PipeY:
// GMSL-A Input Stream: VC0 YUV422_10bit PortB - Output Stream: VC0 YUV422_10bit PortA (D-PHY)

0x04,0x98,0x03,0x13,0x00, // BACKTOP : BACKTOP12 | CSI_OUT_EN (CSI_OUT_EN): CSI output disabled
// Link Initialization for Deserializer
0x04,0x98,0x00,0x10,0x01, // TCTRL : CTRL0 | AUTO_LINK (AUTO_LINK): Disabled | (Default) LINK_CFG (LINK_CFG): 0x1
// Link Initialization for Deserializer
0x04,0x98,0x00,0x01,0x02, // DEV : REG1 | (Default) DIS_REM_CC (GMSL Link A I2C Port 0): Enabled
0x04,0x98,0x00,0x03,0x57, // DEV : REG3 | DIS_REM_CC_B (GMSL Link B I2C Port 0): Disabled
0x00,0x01, // Warning: The actual recommended delay is 5 usec.
// Video Transmit Configuration for Serializer(s)
0x04,0x84,0x00,0x02,0x03, // DEV : REG2 | VID_TX_EN_Z (VID_TX_EN_Z): Disabled

//  
// INSTRUCTIONS FOR GMSL-A SERIALIZER MAX96717
//  


0x04,0x84,0x03,0x30,0x00, // MIPI_RX : MIPI_RX0 | (Default) phy_config (Port Configuration): 1x4
0x04,0x84,0x03,0x83,0x80, // MIPI_RX_EXT : EXT11 | (Default) Tun_Mode (Tunnel Mode): Enabled
0x04,0x84,0x03,0x31,0x10, // MIPI_RX : MIPI_RX1 | ctrl1_num_lanes (Port B - Lane Count): 2
0x04,0x84,0x03,0x32,0xE0, // MIPI_RX : MIPI_RX2 | (Default) phy1_lane_map (Lane Map - PHY1 D0): Lane 2 | (Default) phy1_lane_map (Lane Map - PHY1 D1): Lane 3
0x04,0x84,0x03,0x33,0x04, // MIPI_RX : MIPI_RX3 | (Default) phy2_lane_map (Lane Map - PHY2 D0): Lane 0 | (Default) phy2_lane_map (Lane Map - PHY2 D1): Lane 1
0x04,0x84,0x03,0x34,0x00, // MIPI_RX : MIPI_RX4 | (Default) phy1_pol_map (Polarity - PHY1 Lane 0): Normal | (Default) phy1_pol_map (Polarity - PHY1 Lane 1): Normal
0x04,0x84,0x03,0x35,0x00, // MIPI_RX : MIPI_RX5 | (Default) phy2_pol_map (Polarity - PHY2 Lane 0): Normal | (Default) phy2_pol_map (Polarity - PHY2 Lane 1): Normal | (Default) phy2_pol_map (Polarity - PHY2 Clock Lane): Normal
// Controller to Pipe Mapping Configuration
0x04,0x84,0x03,0x08,0x64, // FRONTTOP : FRONTTOP_0 | (Default) CLK_SELZ (CLK_SELZ): Port B | (Default) START_PORTB (START_PORTB): Enabled
0x04,0x84,0x03,0x11,0x40, // FRONTTOP : FRONTTOP_9 | (Default) START_PORTBZ (START_PORTBZ): Start Video
0x04,0x84,0x03,0x15,0x00, // (Default)  (independent_vs_mode): Disabled
// Pipe Configuration Configuration
0x04,0x84,0x00,0x5B,0x02, // CFGV__VIDEO_Z : TX3 | (Default) TX_STR_SEL (TX_STR_SEL Pipe Z): 0x2

// INSTRUCTIONS FOR DESERIALIZER MAX96716A

// Video Pipes And Routing Configuration
0x04,0x98,0x01,0x61,0x32, // (Default)  (STR_SELY): Link A Stream Id 2
// Double Mode Configuration
// MIPI DPHY Configuration
0x04,0x98,0x03,0x30,0x04, // (Default)  (Port Configuration): 2 (1x4)
0x04,0x98,0x04,0x74,0x09, //  (Port A Tunnel Mode): Enabled
0x04,0x98,0x04,0x4A,0x50, //  (Port A - Lane Count): 2
0x04,0x98,0x03,0x33,0x4E, // (Default)  (Lane Map - PHY0 D0): Lane 2 | (Default)  (Lane Map - PHY0 D1): Lane 3 | (Default)  (Lane Map - PHY1 D0): Lane 0 | (Default)  (Lane Map - PHY1 D1): Lane 1
0x04,0x98,0x03,0x35,0x00, // (Default)  (Polarity - PHY0 Lane 0): Normal | (Default)  (Polarity - PHY0 Lane 1): Normal | (Default)  (Polarity - PHY1 Lane 0): Normal | (Default)  (Polarity - PHY1 Lane 1): Normal | (Default)  (Polarity - PHY1 Clock Lane): Normal
// This is to set predefined (coarse) CSI output frequency
// CSI Phy 1 is 1000 Mbps/lane.
0x04,0x98,0x1D,0x00,0xF4,
0x04,0x98,0x03,0x20,0x2A,
0x04,0x98,0x1D,0x00,0xF5,
0x04,0x98,0x03,0x32,0x34, //  (phy_Stdby_2): Put PHY2 in standby mode |  (phy_Stdby_3): Put PHY3 in standby mode
0x04,0x98,0x03,0x13,0x02, //  (CSI_OUT_EN): CSI output enabled
0x04,0x84,0x00,0x02,0x43, // DEV : REG2 | VID_TX_EN_Z (VID_TX_EN_Z): Enabled

0x04,0x84,0x02,0xBF,0xA0,
0x04,0x84,0x02,0xBE,0x00,
0x00,0x01,
0x04,0x84,0x02,0xBE,0x10,
