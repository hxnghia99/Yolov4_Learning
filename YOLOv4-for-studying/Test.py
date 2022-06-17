#===============================================================#
#                                                               #
#   File name   : Test.py                                       #
#   Author      : hxnghia99                                     #
#   Created date: May 24th, 2022                                #
#   GitHub      : https://github.com/hxnghia99/Yolov4_Learning  #
#   Description : YOLOv4 image prediction testing               #
#                                                               #
#===============================================================#


# ##############################################################################################
# """
# INFERENCE TIME
# """
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from YOLOv4_utils import *
# from YOLOv4_config import *

# # IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/kite.jpg"
# IMAGE_PATH = "./YOLOv4-for-studying/IMAGES/lg_street.jpg"
# yolo = Load_YOLOv4_Model()
# detect_image(yolo, IMAGE_PATH, show=True, save=False, CLASSES_PATH=YOLO_CLASS_PATH)

# ##############################################################################################

# print('a')




# import os
# from YOLOv4_config import *
# from YOLOv4_utils import *
# # text_by_line = "E:/dataset/TOTAL/test\images\cyclist_223_171_53_30_11000_2020-10-27-15-17-46-000000.jpg 12,120,69,181,0 80,7,128,27,0 57,5,87,25,0 1069,208,1124,274,2"
# # text_by_line = "E:/dataset/TOTAL/train\images\c1_2020-10-292020-10-29-12-16-28-000138.jpg 762,114,782,176,1 785,120,804,180,1 676,77,692,120,1 651,64,663,109,1 663,71,677,113,1 327,31,342,67,1 364,119,382,183,1 618,76,639,124,1 320,191,344,264,1 411,0,421,23,1 611,74,631,122,1 359,0,366,17,1 282,4,292,32,1 628,58,642,101,1 343,0,353,19,1 208,598,260,647,1 234,298,279,391,1 367,115,395,168,1 369,0,379,16,1 268,6,277,33,1 256,17,269,48,1 904,24,920,47,1 920,26,928,48,1 407,0,415,22,1"
# # text_by_line = "E:/dataset/TOTAL/train\images\\frame_20210422_072856_00116_51.jpg 1577,130,1822,263,0 184,0,260,58,0 38,0,118,77,0 381,159,421,201,1 423,152,463,198,1 111,977,176,1049,2"
# # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-test-dev/images/0000006_01111_d_0000003.jpg 17,155,57,208,1 333,9,556,113,6 124,105,418,248,6 515,330,597,435,4 575,322,669,446,4 633,439,806,530,4 815,424,911,552,4 919,435,996,563,4 1008,436,1088,580,4 1089,443,1183,586,4 1106,292,1175,403,4 961,307,1083,408,4 990,151,1073,217,4 965,97,1087,154,4 181,652,409,758,4 1327,25,1350,54,10 1337,16,1349,48,2"
# # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images/0000335_00785_d_0000047.jpg 632,282,678,330,4 738,611,829,712,4 184,354,253,424,4 531,153,561,191,4 494,128,523,157,4 496,92,521,118,4 611,59,634,82,4 503,42,523,61,4 806,23,829,41,4 838,26,859,41,4 814,15,837,28,4 361,88,387,115,4 31,555,144,663,4 191,186,293,229,4 104,422,145,453,10 189,289,224,309,10 197,282,232,306,10 22,392,61,416,10 29,377,70,408,10 716,41,721,57,1 713,37,719,51,1 706,17,710,33,1 714,21,720,33,1 572,4,587,15,4 488,38,494,51,2 620,14,625,22,2 621,10,625,19,2 487,46,495,54,10 492,27,497,33,10 492,21,497,32,2 354,83,361,105,1 344,80,353,104,1 373,732,404,764,2 383,718,407,759,2"

# # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-val/images/0000295_02400_d_0000033.jpg 598,601,645,657,4 660,545,704,595,4 595,512,637,557,4 517,603,568,657,4 676,624,724,686,4 738,589,790,650,4 805,568,854,621,4 750,507,795,550,4 683,495,727,535,4 668,414,738,447,4 578,405,646,435,4 459,379,521,411,4 426,372,460,398,4 395,350,420,372,4 776,423,845,460,4 763,364,803,385,4 643,322,665,343,4 10,302,49,318,4 168,311,212,327,4 98,387,167,410,4 0,473,79,509,4 5,456,96,489,4 48,449,121,479,4 55,439,134,466,4 87,430,156,453,4 93,420,172,446,4 105,414,172,433,4 93,358,141,381,4 0,325,88,364,9 72,339,117,360,4 422,356,446,380,4 538,393,601,415,4 633,402,690,429,4 865,443,932,472,4 925,405,1105,480,9 993,462,1068,498,4 1113,473,1204,512,4 1179,466,1263,502,4 1251,502,1348,541,4 1130,519,1228,563,4 1244,537,1357,590,4 824,631,973,765,9 754,291,791,304,4 730,285,764,301,4 477,330,499,350,4 448,318,469,335,4 543,315,560,332,4 511,314,530,330,4 487,299,503,312,4 459,299,476,314,4 438,288,454,302,4 632,305,650,321,4 611,285,628,299,4 664,297,681,310,4 607,253,620,263,4 588,251,599,259,4 638,265,652,275,4 540,219,548,227,4 604,235,616,243,4 592,227,603,236,4 545,306,562,321,4 545,299,560,311,4 540,286,556,305,4 515,306,532,321,4 514,296,530,311,4 516,291,532,304,4 514,280,527,294,4 515,273,527,286,4 488,290,505,303,4 490,282,504,296,4 489,273,504,289,4 494,266,506,280,4 494,259,506,271,4 464,289,478,301,4 467,283,480,295,4 469,274,484,286,4 473,269,484,280,4 474,263,487,276,4 477,259,486,270,4 476,251,489,264,4 479,243,489,255,4 480,236,489,249,4 478,219,493,240,9 458,233,473,256,9 461,218,476,239,9 442,279,458,292,4 446,274,459,286,4 452,268,464,280,4 423,310,442,326,4 415,319,433,344,5 453,256,466,274,5 583,222,595,228,5 586,217,595,224,4 576,214,584,222,4 666,289,682,300,4 661,280,674,294,5 819,514,834,545,1 869,536,881,572,1 861,524,871,553,1 991,533,1004,567,1 985,501,996,533,1 976,503,985,533,1 1077,591,1090,630,1 1098,567,1116,608,1 1130,577,1143,614,1 1154,573,1172,611,1 1256,581,1276,615,1 1328,584,1341,620,1 1224,608,1237,647,1 1187,627,1202,665,1 1022,494,1032,521,1 1011,481,1024,514,1 999,490,1010,517,1 976,469,983,496,1 792,472,804,495,1 824,474,836,495,1 832,488,845,511,1 863,462,874,486,1 941,508,957,535,1 918,523,933,548,1 912,503,924,531,1 901,527,908,557,1 897,519,906,553,1 885,529,895,566,1 1016,690,1031,734,1 1048,711,1068,754,1 1065,694,1089,736,1 1092,687,1111,731,1 1110,732,1128,765,1 1061,658,1078,695,1 1027,664,1048,702,1 993,656,1010,693,1 1021,625,1039,659,1 990,628,1006,663,1 979,607,996,637,1 987,591,1003,624,1 981,580,998,611,1 980,556,992,586,1 947,561,962,593,1 966,531,980,562,1 1051,549,1060,589,1 1037,541,1053,578,1 1041,526,1053,554,1 1099,537,1113,575,1 1108,555,1123,592,1 1111,549,1124,580,1 1116,540,1130,575,1 1126,553,1139,586,1 1137,550,1149,586,1 1159,541,1174,578,1 1178,537,1191,573,1 1191,547,1205,583,1 1208,543,1219,577,1 1224,513,1237,546,1 1309,645,1333,685,1 1202,702,1222,753,1 1309,732,1329,765,1 1153,624,1174,660,1 1177,599,1193,640,1 1192,608,1206,640,1 1207,606,1219,648,1 1200,624,1211,667,1 1121,527,1132,554,1 1121,518,1130,547,1 1112,518,1123,549,1 1104,518,1116,545,1 1097,509,1110,541,1 1089,500,1102,530,1 1093,489,1103,517,1 1079,483,1089,513,1 1073,488,1085,519,1 1069,481,1078,510,1 1066,495,1075,522,1 1055,499,1067,529,1 1038,504,1050,536,1 1046,501,1057,533,1 1031,486,1041,517,1 1036,479,1047,506,1 1045,480,1054,506,1 1056,484,1063,511,1 982,485,991,510,1 1061,542,1090,570,3 1012,701,1030,743,3 1029,676,1045,712,3 1057,669,1080,709,3 996,666,1009,702,3 988,639,1002,674,3 946,602,963,635,3 977,564,991,592,3 963,538,977,566,3 945,573,963,598,3 825,498,845,517,3 865,489,890,508,3 856,472,879,488,10 790,479,806,504,10 917,531,934,556,10 942,519,959,542,10 964,540,976,567,3 1043,579,1062,610,10 1080,647,1116,671,3 1072,628,1108,652,3 970,680,990,713,1 964,668,982,703,1 1280,564,1298,592,3 1303,568,1318,594,3 1310,563,1327,594,3 1326,566,1340,596,3 1220,550,1237,578,3 1145,560,1181,582,3 873,511,894,533,3 918,419,927,442,1 917,382,925,404,1 1064,382,1072,406,1 1122,378,1127,399,1 1127,378,1131,399,1 1174,388,1183,411,1 1261,382,1269,403,1 1208,386,1216,406,1 1132,372,1140,394,1 1129,379,1135,399,1 1192,403,1200,424,1 1186,393,1196,414,1 1114,359,1123,380,1 1106,358,1112,377,1 1110,356,1117,375,1 1170,370,1181,382,2 1130,351,1140,372,1 1010,335,1017,352,1 1004,362,1014,377,2 987,369,998,391,1 907,352,915,373,1 877,350,885,369,1 830,377,840,400,1 806,365,814,384,1 912,416,919,443,1 913,403,920,424,1 937,388,945,411,1 985,391,994,417,1 1021,388,1030,417,1 1042,401,1051,427,1 966,373,972,393,1 957,372,965,392,1 956,366,965,390,1 882,372,888,392,1 858,359,864,380,1 808,325,814,343,1 787,343,793,357,1 722,353,731,369,1 690,312,697,328,1 699,323,706,336,1 749,339,754,355,1 762,332,766,349,1 1020,633,1037,662,10 1027,676,1043,711,3 1068,713,1083,746,3 1057,672,1079,705,3 1086,697,1110,740,3 1046,721,1070,760,10 1146,631,1177,667,10 1087,574,1095,616,1 67,658,93,706,1 195,512,212,548,1 181,524,193,559,1 177,518,191,550,1 123,538,140,574,1 0,433,13,462,1 149,506,181,530,3 49,505,82,534,10 26,485,35,517,1 0,492,16,525,1 10,544,30,565,2 0,388,11,412,1 35,367,44,389,1 183,369,191,388,1 204,362,213,380,1 224,383,235,401,1 253,379,261,396,1 293,436,305,458,1 340,365,350,385,1 350,359,357,374,1 305,353,312,369,1 262,348,270,367,1 296,340,304,357,1 286,340,293,356,1 286,348,293,364,1 309,343,317,359,1 391,310,397,328,1 372,328,377,344,1 376,328,384,344,1 377,338,383,357,1 368,335,374,350,1 382,337,389,356,1 210,336,217,355,1 221,334,228,350,1 216,328,221,346,1 246,314,252,332,1 258,300,262,314,1 264,300,270,313,1 229,336,235,355,1 236,336,241,354,1 239,335,245,354,1 243,337,249,355,1 248,336,254,352,1 253,335,259,356,1 257,337,264,355,1 254,357,276,368,3 304,362,312,374,10 294,347,302,359,10 332,361,339,374,10 352,367,358,380,10 335,375,354,385,3 348,343,357,355,10 352,328,359,340,10 284,314,300,322,3 289,307,295,323,1 292,444,306,466,10 153,498,167,525,1"


# # text_by_line = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-test-dev/images/9999938_00000_d_0000207.jpg 76,14,82,21,1 39,54,54,75,4 98,40,103,49,1 136,32,164,47,4 179,21,207,33,4 76,110,82,118,1 73,131,79,139,1 45,185,62,208,4 61,175,80,203,4 96,165,116,192,4 131,154,149,178,4 149,152,165,173,4 165,145,182,171,4 183,141,199,167,4 214,130,233,155,4 264,114,281,138,4 296,103,314,130,4 207,221,225,247,4 240,207,258,234,4 256,199,275,228,4 273,194,291,221,4 325,183,343,207,4 288,186,310,218,5 75,153,105,172,5 0,136,11,177,9 352,0,355,10,1 356,0,360,8,1 306,74,310,82,1 298,98,302,104,1 265,169,285,196,4 249,176,268,203,4 262,272,280,296,4 208,287,228,315,4 131,315,151,343,4 310,300,313,308,1 309,314,312,322,1 327,324,332,332,1 186,382,190,391,1 187,388,194,398,1 165,381,169,388,1 221,332,235,349,5 287,342,302,364,5 280,413,286,422,1 274,435,280,445,1 290,421,297,430,1 304,411,308,419,1 309,406,314,417,1 306,408,310,419,1 287,435,294,444,1 282,438,286,447,1 292,499,296,507,1 289,515,293,525,1 298,514,303,523,1 309,511,313,521,1 297,526,301,534,1 282,539,289,549,1 296,536,302,546,1 309,529,314,539,1 326,524,330,534,1 349,546,358,553,1 345,538,352,547,1 350,528,355,536,1 287,510,292,518,1 305,499,311,509,1 306,521,312,533,1 337,532,341,541,1 277,538,283,546,1 251,583,257,590,1 269,589,281,594,2 286,615,290,623,1 293,613,299,619,1 352,611,358,619,1 331,624,338,631,1 303,684,310,690,1 368,731,379,739,1 492,703,497,712,1 395,779,400,785,2 387,750,393,755,2 389,757,395,762,2 391,716,397,722,2 387,721,391,727,2 376,719,381,724,2 382,724,388,731,2 386,711,392,718,2 378,704,383,712,2 390,680,397,687,2 383,679,389,688,2 458,649,463,656,2 451,652,458,660,2 572,581,579,589,2 469,560,474,568,2 484,565,488,569,2 496,561,500,566,2 517,540,523,547,1 599,560,603,569,1 380,557,385,565,1 400,542,406,550,1 412,537,416,547,1 409,545,413,552,1 421,542,425,549,1 414,543,418,551,1 495,501,501,509,1 488,504,492,513,1 489,515,492,522,1 493,509,498,518,1 501,503,506,513,1 585,463,591,468,1 505,433,509,438,1 524,435,528,440,1 403,404,501,466,9 460,475,464,484,1 456,480,461,489,1 484,302,574,357,9 542,331,579,355,5 369,393,412,424,5 500,346,540,373,5 463,366,496,388,4 394,296,412,326,5 557,255,562,268,1 404,156,426,182,4 379,137,396,159,4 348,149,366,174,4 442,117,460,140,4 496,118,518,147,5 535,29,556,55,4 564,23,580,44,4 579,17,597,40,4 435,63,452,88,4 389,74,406,99,4 500,96,518,119,4 517,93,535,114,4 542,108,562,134,4 585,93,607,118,4 487,103,506,126,4 498,189,515,214,4 514,181,535,211,4 529,175,550,201,4 465,200,483,226,4 572,73,593,99,5 459,27,506,56,5 504,16,531,33,5 556,350,561,360,1 544,353,549,363,1 415,46,445,61,4 378,62,410,77,4 359,22,363,31,1 377,2,381,11,1 380,0,384,8,1 395,10,401,19,1 411,26,416,34,1 406,31,409,39,1 403,34,406,41,1 438,18,441,25,1 445,7,449,15,1 454,12,459,18,1 471,0,475,9,1 477,8,482,19,1 489,9,493,15,1 536,7,539,16,1 441,0,445,7,1 400,32,403,40,1 443,7,447,17,1 454,8,458,16,1 492,17,496,25,1 457,30,461,39,1 410,41,414,47,1 408,45,411,54,1 414,43,417,50,1 365,47,368,55,1 632,1,651,20,4 603,64,622,86,4 716,113,739,139,4 592,12,610,35,4 636,84,653,104,4 740,33,772,72,9 734,60,763,100,6 774,118,807,159,9 790,97,823,141,9 828,87,849,111,4 755,97,759,106,1 765,98,769,106,1 761,88,766,96,1 707,140,723,160,4 661,154,673,175,5 647,163,660,182,5 545,202,560,220,4 629,186,643,204,4 615,177,626,193,4 646,184,658,198,4 691,257,721,285,4 666,192,669,203,1 676,188,679,196,1 752,177,764,184,10 738,224,743,231,1 743,212,746,219,1 735,221,737,230,1 739,285,742,292,1 733,286,739,295,1 712,316,716,325,1 603,313,606,325,1 590,316,623,337,4 684,236,688,248,1 725,414,730,423,1 715,420,718,427,1 718,417,721,425,1 719,408,723,416,1 775,492,780,501,1 685,772,689,777,2 690,772,695,779,2 699,762,703,770,2 730,750,735,755,2 712,757,716,762,2 733,745,737,749,2 729,744,732,748,2 724,750,729,757,2 719,757,725,765,1 696,769,699,775,1 742,740,745,752,1 738,745,741,754,1 789,14,793,25,1 840,32,844,39,1 844,29,847,37,1 854,56,858,68,1 858,68,862,75,1 865,60,869,69,1 877,65,880,73,1 864,55,867,61,1 859,18,863,26,2 886,96,889,104,1 893,92,898,99,1 901,112,904,119,1 904,113,907,120,1 914,132,917,140,1 919,138,922,146,1 924,138,926,146,1 918,133,921,140,1 925,132,927,141,1 862,112,866,118,1 857,103,860,111,1 876,118,879,126,1 873,121,875,125,1 879,118,884,125,1 892,127,896,136,1 888,128,891,137,1 884,145,887,152,1 903,129,906,138,1 906,134,909,142,1 909,141,912,148,1 943,146,947,154,1 944,152,947,160,1 947,156,950,166,1 953,149,956,157,1 893,156,898,163,1 898,164,902,173,1 892,161,896,169,1 884,164,888,174,1 877,160,881,169,1 873,161,876,169,1 923,196,928,205,1 918,202,922,210,1 949,81,953,90,1 952,97,957,104,1 925,0,940,10,4 941,16,969,51,6 953,165,956,169,2 958,164,960,170,2 976,162,980,170,1 913,201,916,208,1 865,236,869,245,1 854,236,858,244,1 851,238,854,244,1 845,256,850,264,1 905,202,909,208,1 910,205,913,212,1 854,171,867,196,4 827,199,857,230,5 790,222,823,244,5 748,234,792,262,5 843,218,846,227,1 838,229,843,240,1 838,222,841,229,1 823,232,828,241,1 795,244,799,253,1 820,274,826,287,1 835,279,840,291,1 796,313,802,321,1 885,271,889,279,1 916,255,919,260,1 945,216,949,226,1 951,213,954,220,1 931,220,935,223,1 935,225,939,230,1 946,233,950,244,1 866,311,871,318,1 903,313,909,322,1 856,320,861,329,1 847,318,851,325,1 978,298,982,306,1 827,368,830,376,1 946,377,951,384,1 952,376,956,384,1 950,373,953,381,1 836,430,840,438,1 875,435,880,445,1 894,512,900,521,1 835,442,840,448,2 840,449,846,457,2 940,556,944,565,1 815,485,820,493,1 803,526,807,535,1 808,532,810,539,1 816,569,820,576,1 810,563,813,569,1 811,575,815,582,1 873,561,877,570,1 868,557,872,566,1 835,706,839,712,1 852,687,858,695,2 863,681,870,688,2 828,699,832,706,2 834,692,838,700,2 840,694,846,701,2 803,712,807,719,2 788,718,793,727,2 793,716,798,726,2 798,712,804,722,2 778,721,784,729,2 1116,680,1123,687,2 1123,686,1128,694,2 1113,676,1118,682,2 1034,611,1040,618,2 1039,616,1044,622,2 1057,643,1062,648,2 1062,648,1067,653,2 1069,657,1075,663,2 1002,577,1007,586,2 1008,579,1013,586,2 1171,706,1180,715,1 1183,708,1190,715,1 1188,713,1198,720,1 1183,727,1192,734,1 1170,730,1180,737,1 1171,724,1181,730,1 1190,722,1197,729,1 1179,722,1184,728,1 1180,718,1184,723,1 1127,648,1134,656,1 1151,627,1159,634,1 1122,609,1128,616,1 1142,645,1150,652,1 1139,648,1147,656,1 1121,655,1131,662,1 1206,609,1216,615,1 1143,622,1151,627,1 1148,624,1153,633,1 1148,633,1155,639,1 1120,611,1124,617,1 1070,614,1075,622,1 1094,552,1103,559,1 1103,544,1111,551,1 1096,521,1100,532,1 1123,531,1126,541,1 1108,535,1116,544,1 1055,520,1060,529,1 1126,530,1132,538,1 1117,533,1121,542,1 1202,430,1209,441,1 1161,429,1166,441,1 1155,428,1162,438,1 1095,415,1103,423,1 1108,389,1117,400,1 1088,379,1095,390,1 1119,383,1125,392,1 1107,375,1113,383,1 1023,390,1028,400,1 1067,393,1071,399,1 1072,394,1077,400,1 1203,391,1215,398,1 1116,348,1122,357,1 1124,351,1127,359,1 1125,348,1131,356,1 1179,357,1186,365,2 1177,356,1185,360,2 1174,352,1183,356,2 1171,349,1181,356,2 1175,345,1181,351,2 1167,314,1172,319,2 1157,311,1163,317,2 1124,259,1126,264,2 1176,223,1182,231,1 1183,213,1189,220,1 1158,202,1164,212,1 1156,189,1163,199,1 1122,146,1128,156,1 1019,149,1025,161,1 1006,195,1011,202,1 1016,185,1019,191,1 1058,185,1063,193,1 1083,184,1088,192,1 1091,166,1096,171,1 985,159,989,165,2 988,164,992,170,2 1005,169,1008,174,2 1052,176,1055,182,1 1050,187,1053,193,1 1058,177,1061,182,1 1072,176,1077,181,1 1078,182,1081,189,1 1072,184,1076,188,1 1079,174,1082,182,1 1047,185,1049,190,2 1051,182,1056,186,2 1038,175,1042,182,2 1029,180,1034,188,2 1020,177,1027,184,2 1026,172,1029,176,2 1030,175,1034,181,2 1011,174,1015,180,2 999,173,1004,180,2 1000,181,1004,186,2 1007,178,1013,187,2 1004,175,1009,181,2 1126,166,1132,172,2 1005,42,1009,51,1 1008,65,1013,74,1 1109,95,1114,104,1 1111,90,1115,100,1 1106,84,1110,92,1 1107,88,1110,96,1 1033,11,1037,19,1 1027,12,1031,20,1 1030,13,1035,22,1 1325,59,1330,69,1 1338,62,1341,72,1 1343,63,1347,71,1 1232,278,1241,288,1 1284,348,1289,357,1 1277,345,1285,355,1 1295,377,1301,385,1 1302,376,1307,382,1 1133,146,1136,155,1 1335,54,1338,62,1 1331,57,1333,65,1 1358,283,1364,292,1 1250,452,1257,462,1 1259,459,1263,469,1 1276,501,1282,510,1 1282,506,1289,514,1 1266,557,1274,565,1 1302,560,1309,565,1 1343,557,1353,565,1 1296,511,1306,517,2 1305,513,1310,521,2 1235,621,1243,629,1 1244,621,1248,630,1 1385,670,1395,675,1 1369,630,1376,635,1 1356,745,1360,754,1 1362,741,1367,748,1 1250,614,1254,624,1 1246,617,1249,625,1"
# text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999982_00000_d_0000034.jpg 1168,406,1193,435,5 590,330,657,352,0'


# text = text_by_line.split()


# bboxes = []
# for t in text:
#     if not t.replace(',', '').isnumeric():
#         temp_path   = os.path.relpath(t, RELATIVE_PATH)
#         temp_path   = os.path.join(PREFIX_PATH, temp_path)
#         image_path  = temp_path.replace('\\','/')
#     else:
#         t = list(map(int, t.split(',')))
#         bboxes.append(t)
# bboxes = np.array(bboxes)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 0:
#     print(f'GPUs {gpus}')
#     try: tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError: pass

# yolo = Load_YOLOv4_Model()
# pred_image = detect_image(yolo, image_path, show=False, show_label=False, save=False, CLASSES_PATH=YOLO_CLASS_PATH, score_threshold=VALIDATE_SCORE_THRESHOLD)

# image = cv2.imread(image_path)
# image = draw_bbox(image, bboxes, YOLO_CLASS_PATH, show_label=False)
# cv2.imshow('truth', cv2.resize(image,(1280, 720)))
# cv2.imshow("prediction", cv2.resize(pred_image,(1280, 720)))

# if cv2.waitKey() == 'q':
#     pass



import os
from YOLOv4_config import *
from YOLOv4_utils import *



# text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999982_00000_d_0000034.jpg 1168,406,1193,435,5 590,330,657,352,0'
# text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999999_00650_d_0000295.jpg 147,344,173,359,4 177,326,213,344,4 188,303,221,323,4 202,245,232,258,4 283,232,308,243,4 236,207,248,228,5 204,203,227,217,4 273,195,294,205,4 265,178,286,188,5 209,162,229,170,4 199,151,214,159,5 200,146,219,152,4 258,206,260,213,1 298,89,304,95,4 200,93,214,98,4 204,87,215,92,4 237,50,245,53,4 233,45,242,49,4 231,37,239,39,4 230,27,237,30,4 424,91,436,98,4 383,58,393,63,4 342,25,351,32,6 382,48,395,58,6 197,64,207,68,4 198,47,205,59,9 227,49,234,60,9'
text_by_line = './YOLOv4-for-studying/dataset/Visdrone_DATASET/VisDrone2019-DET-train/images/9999965_00000_d_0000023.jpg 682,661,765,695,3 904,567,934,644,3 958,484,1023,558,3 979,286,1013,359,3 818,554,890,584,3 799,494,891,531,3 821,455,890,484,3 813,402,890,432,3 816,358,893,388,3 817,309,892,338,3 825,267,896,299,3 815,227,885,259,3 697,152,773,193,3 810,49,890,86,3 805,10,885,42,3 988,83,1019,156,4 824,190,896,219,4 331,25,369,103,4 800,138,898,183,5 689,287,774,326,3 700,344,777,379,3 697,402,777,432,3 697,502,774,537,3 701,550,765,579,4 544,389,598,586,8 314,625,346,702,3 318,535,351,609,3 321,438,357,515,3 329,333,363,405,3 329,227,356,302,3 317,119,364,213,5 308,733,346,786,3 633,77,647,117,9 633,87,648,106,1 954,69,969,86,0 1003,462,1018,474,0 1041,445,1055,456,0 917,133,928,146,0 740,740,753,761,0 1016,681,1028,695,0 1053,471,1062,483,0'


text = text_by_line.split()
bboxes = []
for t in text:
    if not t.replace(',', '').isnumeric():
        temp_path   = os.path.relpath(t, RELATIVE_PATH)
        temp_path   = os.path.join(PREFIX_PATH, temp_path)
        image_path  = temp_path.replace('\\','/')
    else:
        t = list(map(int, t.split(',')))
        bboxes.append(t)
image = cv2.imread(image_path)
bboxes = np.array(bboxes)

# image = draw_bbox(image, bboxes, YOLO_CLASS_PATH, show_label=False)
# cv2.imshow('truth', cv2.resize(image,(1280, 720)))
# if cv2.waitKey() == 'q':
#     pass



#Create format of each sliced_image object including 4 attributes
class SlicedImage:
    def __init__(   self,                           
                    image: np.ndarray,              #cv2 image in Numpy array
                    bboxes,                         #List of [4 coordinates, class_idx]
                    starting_point,                 #[xmin, ymin]
                    predictions = None):                   #List of [4 coordinates, score, classs_idx]
        self.image = image
        self.bboxes = bboxes
        self. starting_point = starting_point
        self.predictions = predictions

#Create format of object processed for each original image
class Original_Image_Into_Sliced_Images:
    def __init__(self, original_image=None, original_bboxes=None):              #inputs as original image and all gt bboxes inside that image
        #Setting for slicing original image into set of sliced images
        self.original_image = original_image
        self.original_bboxes = original_bboxes
        self.original_image_height = self.original_image.shape[0]
        self.original_image_width = self.original_image.shape[1]
        self.sliced_image_size = SLICED_IMAGE_SIZE
        self.overlap_ratio = OVERLAP_RATIO
        self.min_area_ratio = MIN_AREA_RATIO
        
        #List of sliced images
        self.sliced_image_list = []







        """ Test """
        self.slice_image(self.original_image, self.original_bboxes, *self.sliced_image_size, *self.overlap_ratio, self.min_area_ratio)


    #Get the bbox coordinate of sliced images in origional image
    def get_sliced_image_coordinates(   self,
                                        image_width: int,                   
                                        image_height: int,
                                        slice_width: int,
                                        slice_height: int,
                                        overlap_width_ratio: float,
                                        overlap_height_ratio: float):  
        sliced_image_coordinates = []
        x_overlap = int(overlap_width_ratio * slice_width)
        y_overlap = int(overlap_height_ratio * slice_height)
        #Run in y-axis, at each value y, calculate x
        y_max = y_min = 0
        while y_max < image_height:    
            #update new ymax for this iterative
            y_max = y_min + slice_height
            #run in x-axis, at each value (xmin,xmax), save the patch coordinates
            x_min = x_max = 0
            while x_max < image_width:
                #update new xmax for this iterative
                x_max = x_min + slice_width
                #if the patch coordinates is outside original image, cut at the borders to inside area inversely
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)   
                    ymin = max(0, ymax - slice_height)
                    sliced_image_coordinates.append([xmin, ymin, xmax, ymax])
                else:
                    sliced_image_coordinates.append([x_min, y_min, x_max, y_max])
                #update new xmin for next iterative
                x_min = x_max - x_overlap
            #update new ymin for next iterative
            y_min = y_max - y_overlap
        return sliced_image_coordinates


    #check if gt_coordinates is inside the sliced image
    def check_gt_coordinates_inside_slice(  self, 
                                            gt_coordinates,                 #format [xmin, ymin, xmax, ymax]                   
                                            sliced_image_coordinates):      #format [xmin, ymin, xmax, ymax]                   
        if gt_coordinates[0] >= sliced_image_coordinates[2]:    #if gt is left to sliced_image
            return False    
        if gt_coordinates[1] >= sliced_image_coordinates[3]:    #if gt is below sliced_image
            return False
        if gt_coordinates[2] <= sliced_image_coordinates[0]:    #if gt is right to sliced_image
            return False        
        if gt_coordinates[3] <= sliced_image_coordinates[1]:    #if gt is above sliced_image
            return False
        return True

    #Tranform gt_bboxes in original image into those in sliced images
    def process_gt_bboxes_to_sliced_image(  self, 
                                            original_gt_bboxes,                         #List of gt bboxes with format [4 coordinates, class_idx]
                                            sliced_image_coordinates,                   #format [xmin, ymin, xmax, ymax]
                                            min_area_ratio):                            #area ratio to remove gt bbox from sliced image
        #Each ground truth bbox is compared to sliced_image_coordinates to create bbox_coordinates inside sliced_image
        sliced_image_gt_bboxes = []
        for original_gt_bbox in original_gt_bboxes:
            if self.check_gt_coordinates_inside_slice(original_gt_bbox[:4], sliced_image_coordinates):
                #Calculate intersection area
                top_left        = np.maximum(original_gt_bbox[:2], sliced_image_coordinates[:2])
                bottom_right    = np.minimum(original_gt_bbox[2:4], sliced_image_coordinates[2:])
                gt_bbox_area = np.multiply.reduce(original_gt_bbox[2:4] - original_gt_bbox[:2])
                intersection_area = np.multiply.reduce(bottom_right - top_left)
                if intersection_area/gt_bbox_area >=min_area_ratio:
                    sliced_image_gt_bbox = np.concatenate([top_left - sliced_image_coordinates[:2], bottom_right - sliced_image_coordinates[:2], np.array([original_gt_bbox[4]])])  #minus starting point
                    sliced_image_gt_bboxes.append(sliced_image_gt_bbox)
        return sliced_image_gt_bboxes


    #slice the original image into objects of class SliceImage
    def slice_image(self, 
                    original_image,                 #original image
                    original_gt_bboxes,             #list of original bboxes with shape [4 coordinates, class_idx]
                    slice_width,                
                    slice_height,
                    overlap_width_ratio,
                    overlap_height_ratio,
                    min_area_ratio):

        original_image_height, original_image_width, _ = original_image.shape
        if not (original_image_width != 0 and original_image_height != 0):
            raise RuntimeError(f"Error from invalid image size: {original_image.shape}")
       
        sliced_image_coordinates_list = self.get_sliced_image_coordinates(*[original_image_width, original_image_height], *[slice_width, slice_height], *[overlap_width_ratio, overlap_height_ratio])
        
        
        number_images = 0
        # iterate over slices
        for sliced_image_coordinates in sliced_image_coordinates_list:
            # count number of sliced images
            number_images += 1
            # Extract starting point of the sliced image
            starting_point = [sliced_image_coordinates[0], sliced_image_coordinates[1]]
            # Extract sliced image
            tl_x, tl_y, br_x, br_y = sliced_image_coordinates
            sliced_image = np.copy(original_image[tl_y:br_y, tl_x:br_x])
            # Extract gt bboxes
            sliced_image_gt_bboxes = self.process_gt_bboxes_to_sliced_image(np.copy(original_gt_bboxes), sliced_image_coordinates, min_area_ratio)

            if len(sliced_image_gt_bboxes) != 0:
                sliced_image_obj = SlicedImage(sliced_image, sliced_image_gt_bboxes, starting_point)
                self.sliced_image_list.append(sliced_image_obj)
                
                
                
                
            #     sliced_image = draw_bbox(sliced_image, sliced_image_gt_bboxes, YOLO_CLASS_PATH, show_label=False)
            # cv2.imshow("test", sliced_image)
            # if cv2.waitKey() == 'q':
            #     pass
            # cv2.destroyAllWindows()
            # print("OK!")




    















#Get the bbox coordinate of slicing patches in origional image
def get_slicing_patch_coordinates(  image_width: int,
                                    image_height: int,
                                    slice_width: int = 512,
                                    slice_height: int = 512,
                                    overlap_width_ratio: int = 0.2,
                                    overlap_height_ratio: int = 0.2):
    slicing_patch_coordinates = []
    x_overlap = int(overlap_width_ratio * slice_width)
    y_overlap = int(overlap_height_ratio * slice_height)
    #Run in y-axis, at each value y, calculate x
    y_max = y_min = 0
    while y_max < image_height:    
        #update new ymax for this iterative
        y_max = y_min + slice_height
        #run in x-axis, at each value (xmin,xmax), save the patch coordinates
        x_min = x_max = 0
        while x_max < image_width:
            #update new xmax for this iterative
            x_max = x_min + slice_width
            #if the patch coordinates is outside original image, cut at the borders to inside area inversely
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)   
                ymin = max(0, ymax - slice_height)
                slicing_patch_coordinates.append([xmin, ymin, xmax, ymax])
            else:
                slicing_patch_coordinates.append([x_min, y_min, x_max, y_max])
            
            #update new xmin for next iterative
            x_min = x_max - x_overlap
        #update new ymin for next iterative
        y_min = y_max - y_overlap
    return slicing_patch_coordinates


def slice_image(original_image: np.ndarray,
                slice_width: int = 512,
                slice_height: int = 512,
                overlap_width_ratio: float = 0.2,
                overlap_height_ratio: float = 0.2,
                min_area_ratio: float = 0.1,):

    image_height, image_width, _ = original_image.shape
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"Error from invalid image size: {image.shape}")
    slicing_patch_coordinates = get_slicing_patch_coordinates(*[image_width, image_height], *[slice_width,slice_height], *[overlap_width_ratio,overlap_height_ratio])

    

    # # init images and annotations lists
    # sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width])

    sliced_image_list = [] #List of SlicedImage
    number_images = 0
    # iterate over slices
    for slice_bbox in slicing_patch_coordinates:
        number_images += 1

        # extract image
        tl_x, tl_y, br_x, br_y = slice_bbox
        sliced_image = image[tl_y:br_y, tl_x:br_x]


        


        """ANNOTATION PROCESSING"""
        # # process annotations if coco_annotations is given
        # if coco_annotation_list is not None:
        #     sliced_coco_annotation_list = process_coco_annotations(coco_annotation_list, slice_bbox, min_area_ratio)
        #  # append coco annotations (if present) to coco image
        # if coco_annotation_list:
        #     for coco_annotation in sliced_coco_annotation_list:
        #         coco_image.add_annotation(coco_annotation)

       

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=sliced_image,
            bboxes=[],
            starting_point=[slice_bbox[0], slice_bbox[1]],
            predictions=[],
        )
        # sliced_image_result.add_sliced_image(sliced_image)

        sliced_image_list.append(slice_image)


    return sliced_image_list










# import numpy as np


# anchor =    np.array([[52, 52],
#                         [28, 17],
#                         [16, 11],
#                         [17, 27],
#                         [ 6,  8],
#                         [10, 18],
#                         [28, 40],
#                         [47, 27],
#                         [96, 75]])
# print(anchor)

# anchor_area = np.multiply.reduce(anchor, axis=-1)
# print(anchor_area)

# anchor_n = []
# while len(anchor):
#     i = np.argmin(anchor_area)
#     anchor_n.append(anchor[i])

#     anchor = np.delete(anchor, i, axis=0)
#     anchor_area = np.delete(anchor_area, i, axis=0)

# anchor_n = np.array(anchor_n)  
# print(anchor_n)