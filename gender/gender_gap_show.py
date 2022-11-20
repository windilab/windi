import pandas as pd
import numpy as np
import seaborn as s
import codecs
import japanize_matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("gender_gap_full.csv", delimiter=",")
print(df.columns.values)
list = df.columns.values
list = list[7:]
print(list)
s = pd.Series(list)

s.to_csv("gender_indicator.csv")

"""
df = pd.read_csv("male_female_estat_.csv", delimiter=",")
print(df.head(10))

df = df[["調査年", "地域", "F110801_非労働力人口（男）【人】", "F110802_非労働力人口（女）【人】"]]
print(df.head(15))

df["調査年"] = df["調査年"].replace("年度", "", regex=True)
df["調査年"] = df["調査年"].astype(int)


print(df)

df.to_csv("gender_non_worker.csv")
"""

'''
ls = [1.2446905804860384e-05, 3.706281414281181e-05, 2.9569876268205658e-05, 3.513270996435084e-05, 2.0376777967565083e-05, 5.1941954136443976e-05, 1.9869652239247362e-05, 1.8904418540679935e-05, 1.774837369063638e-05, 1.9261236153195273e-05, 1.5043442757303473e-05, 2.2616337401657884e-05, 2.9011489618315937e-05, 0.00021243685498088368, 4.7691726892012335e-05, 1.5203479361770667e-05, 2.4027779652342177e-05, 8.532265823568247e-05, 3.557531931091603e-05, 2.339671273692193e-05, 2.126569894565243e-05, 2.656376919419797e-05, 0.00010371338859123715, 1.554204372170159e-05, 9.915310736408219e-06, 6.313942176905476e-05, 2.304659347548334e-05, 4.898263712321525e-05, 2.1969765955045022e-05, 1.5706359478789128e-05, 2.7849228258271537e-05, 2.030506314123114e-05, 2.4160591788930576e-05, 2.7808920878018773e-05, 1.612743830252284e-05, 2.812705059619253e-05, 3.818423874433863e-05, 2.5764381419826542e-05, 3.3504979538743896e-05, 3.0984523352676985e-05, 2.2023659948295538e-05, 4.0502093879012866e-05, 2.545788389135965e-05, 3.478871768734116e-05, 2.7902558982525867e-05, 2.4738203907906008e-05, 2.8643371833157326e-05, 2.963582841507584e-05, 2.639505917467045e-05, 2.7329856575377863e-05, 3.361712433223771e-05, 6.758175890368627e-05, 2.1050095972661116e-05, 5.7019455558300424e-05, 2.6703885810726367e-05, 2.357042007534746e-05, 3.338201122265517e-05, 1.45124966593498e-05, 4.6076665403444256e-05, 1.680388615984185e-05, 3.09030890927914e-05, 1.6165718104722745e-05, 1.7606262047504315e-05, 2.3847957353063023e-05, 2.0119106652499e-05, 2.1538433863445143e-05, 2.538193392202593e-05, 3.373248848628559e-05, 3.458757590224837e-05, 2.3214001235998403e-05, 1.6622824456193458e-05, 2.407240366724637e-05, 2.077311278663733e-05, 4.1875747404046146e-05, 8.773206810583782e-05, 1.60903779780166e-05, 2.5212258981841254e-05, 2.6746199952568874e-05, 2.0184133580504834e-05, 2.0156938048592624e-05, 2.206426302233225e-05, 2.2373241207946287e-05, 1.9025162607027725e-05, 2.158839820343652e-05, 2.6247912878876683e-05, 1.921994495086567e-05, 5.931726410839558e-05, 2.0581209436609936e-05, 5.597209153018992e-05, 2.733624017647622e-05, 2.2589646582112274e-05, 2.006650703800483e-05, 1.5709816537639656e-05, 3.0886650879297666e-05, 3.0853934844233604e-05, 2.42964990963745e-05, 2.603820814756544e-05, 1.659138031560395e-05, 2.5613417731100633e-05, 6.538320011717849e-05, 2.336201581618089e-05, 4.710949889750479e-05, 1.217695004651487e-05, 1.6629895003702602e-05, 4.086971880891165e-05, 1.469160432897329e-05, 2.157584053706195e-05, 3.436667469083349e-05, 1.9704679802710146e-05, 3.2315716600885075e-05, 2.4212207257242887e-05, 2.6579588427579978e-05, 2.44944785708529e-05, 8.008593028041707e-05, 2.1201317596320603e-05, 8.697971121148022e-05, 2.2779689836400856e-05, 1.63378115173272e-05, 6.240439916621612e-05, 1.639062201743407e-05, 2.0389370989300618e-05, 1.993286148488889e-05, 2.052015202902954e-05, 1.2452658676598577e-05, 2.2190864538227214e-05, 4.752679919287098e-05, 3.235421386653385e-05, 1.1217603621722211e-05, 2.5531233905411936e-05, 6.658872111402103e-05, 1.41966952389545e-05, 1.3042054616564778e-05, 1.799596616306973e-05, 5.102460332658932e-05, 2.3508328586218492e-05, 1.5584497346146764e-05, 2.796709152770482e-05, 2.1998611570500613e-05, 2.6133354811260358e-05, 5.152415518263336e-05, 3.328640066661618e-05, 6.182526494643678e-05, 4.29004575978445e-05, 2.6463274637998982e-05, 1.384341874436341e-05, 1.9500675906994558e-05, 1.6585451684121574e-05, 1.661458756098141e-05, 3.96014175991871e-05, 1.9456324869078942e-05, 3.443271086529796e-05, 2.908767960944791e-05, 3.2572819143853374e-05, 3.080318354497332e-05, 5.9129852379736105e-05, 2.584369143172669e-05, 2.3700336400311185e-05, 2.295385433678564e-05, 1.949441234573813e-05, 0.0001047051755568507, 4.7756012498776904e-05, 3.0621426254732896e-05, 3.0330329787574476e-05, 0.00011351280602452872, 5.899811327758896e-05, 4.487559885992942e-05, 1.6058071518658767e-05, 5.332042520584266e-05, 1.4641034081562914e-05, 5.5317846623907034e-05, 4.7633211708035864e-05, 2.0871257681839354e-05, 3.257274538544928e-05, 5.3935874395199195e-05, 4.0932558567526e-05, 2.6101896683267032e-05, 2.2139015189407003e-05, 3.960938122663364e-05, 2.4003884747272092e-05, 1.8993444820794937e-05, 2.1812110866525305e-05, 2.3581327996615714e-05, 5.3441220735573664e-05, 1.676961373663781e-05, 2.3139957970158342e-05, 2.4867696817964187e-05, 2.7155340235561943e-05, 2.7706119373872422e-05, 3.455247506755714e-05, 2.7019267734456333e-05, 1.7863647551860632e-05, 9.997886642212663e-05, 2.5250822199306408e-05, 0.0001148428787810279, 3.125054048752464e-05, 0.00011692922301634906, 2.168951486759629e-05, 2.2779506534899504e-05, 2.1052823435077896e-05, 1.6244950640908765e-05, 2.8620389447510113e-05, 1.7812925277746906e-05, 1.7272610165667503e-05, 0.00013397606686233488, 3.1814947528489024e-05, 3.6314417558942153e-05, 1.6285954577328594e-05, 2.8052451167705763e-05, 5.5018628993227255e-05, 7.120413986321704e-05, 1.3503800238776976e-05, 6.43955935030505e-05, 2.311167766182133e-05, 4.5984967300840574e-05, 1.8179756224752096e-05, 3.914353800576804e-05, 1.9925472861694028e-05, 1.938167404702747e-05, 3.163459378694822e-05, 5.286804520844706e-05, 7.447559685132538e-05, 4.5881534421152585e-05, 2.8626353079075668e-05, 2.1326069362027e-05, 6.313855530865395e-05, 3.205270062364979e-05, 2.9904330105577913e-05, 1.9420200881119258e-05, 2.161116632918686e-05, 5.0404614907408365e-05, 2.236092914833025e-05, 2.0718341053220713e-05, 1.9828734473690482e-05, 4.29682904343005e-05, 1.6827310217828442e-05, 7.64287696891934e-05, 2.617935092006475e-05, 0.0001042555883117718, 2.107195361274513e-05, 1.9584923680051575e-05, 6.500411438925802e-05, 1.3867005845412675e-05, 2.0780933788667847e-05, 2.208912959676814e-05, 1.632713102820282e-05, 3.534614018129684e-05, 4.4298672844551405e-05, 3.1937955245267885e-05, 1.8219114643798333e-05, 4.042566753745181e-05, 2.1394066025218982e-05, 3.0475155411280798e-05, 2.2094619687204717e-05, 1.1710399021917363e-05, 1.9900184880342612e-05, 5.058655802770737e-05, 1.7565795558639022e-05, 1.6905305581611656e-05, 1.700591652819358e-05, 1.8143434470442827e-05, 2.7704236101164628e-05, 2.0069025992862926e-05, 1.3885621275978087e-05, 2.3019836289681458e-05, 3.9495059738868834e-05, 3.145678701375172e-05, 2.2555780025943244e-05, 0.00014328164079342026, 1.6862916058414836e-05, 1.1965457120210789e-05, 1.539748653784363e-05, 1.2602557911493509e-05, 1.824192465221524e-05, 2.1517447506709265e-05, 5.393680241700303e-05, 4.275239851285459e-05, 1.8949979482009925e-05, 1.537331008062505e-05, 2.386234873640221e-05, 4.161507081291984e-05, 2.928001966527251e-05, 2.009499850276921e-05, 3.326050353859295e-05, 2.5846985323862387e-05, 2.716905812363319e-05, 1.8586833784803768e-05, 1.2409766658503327e-05, 1.4274782775881428e-05, 3.48039088329754e-05, 2.397729484307379e-05, 6.559574919298655e-05, 1.3805602706743894e-05, 4.298878996901859e-05, 5.7414752236828244e-05, 3.1297765572817037e-05, 2.245043825003503e-05, 4.010473930337587e-05, 6.436695152726461e-05, 2.134367376604e-05, 1.9745876920973853e-05, 1.7527385063893745e-05, 1.364496076998161e-05, 2.0359546961880572e-05, 0.00010012628589239023, 1.6772717080117183e-05, 2.7374110766044678e-05, 2.586316597984251e-05, 1.775554990660448e-05, 1.6869439679629972e-05, 6.440737993806675e-05, 2.9111685073228824e-05, 5.8750488656534966e-05, 3.252013926664979e-05, 7.040210993837628e-05, 1.6228545854858937e-05, 5.170932055735879e-05, 3.364164417047585e-05, 3.893762147614551e-05, 1.4567494092674697e-05, 3.2363151148403975e-05, 5.529838908039767e-05, 1.912112062905757e-05, 2.033010023959575e-05, 4.069123001080374e-05, 3.633863205633256e-05, 5.312655735617425e-05, 0.00011128862849218227, 2.617225632796997e-05, 5.940191565016269e-05, 7.777675060608654e-05, 1.5672110984871943e-05, 6.551707468323664e-05, 2.1747346036035875e-05, 5.83689238701782e-05, 2.653697712108654e-05, 2.6270324192200887e-05, 1.4282937229260195e-05, 5.4482412430251244e-05, 2.0353398769872578e-05, 2.4834958887124376e-05, 2.1694193368637294e-05, 6.155886665767586e-05, 2.4893358796819204e-05, 3.3678952977593063e-05, 1.5402916800784252e-05, 1.516649772886937e-05, 2.3948106552903526e-05, 2.7448308419114374e-05, 1.984288374315055e-05, 4.252836411298082e-05, 2.8459072441030933e-05, 2.2649427244994598e-05, 2.452969272112758e-05, 4.254232238004031e-05, 4.3871487929623904e-05, 4.104918842762352e-05, 3.679797455012689e-05, 1.613944931205251e-05, 2.973916794791191e-05, 2.3720129442400866e-05, 1.3389125907366911e-05, 4.525853206945841e-05, 1.807019172228799e-05, 5.39457138122957e-05, 2.4299610580910385e-05, 1.68386624791488e-05, 3.316321626077552e-05, 2.717161304787329e-05, 3.182934151678886e-05, 1.790869335866129e-05, 2.2299390618706724e-05, 3.169409557410604e-05, 1.978329516287013e-05, 2.6398572936561804e-05, 2.0494849841245083e-05, 2.8077211681810192e-05, 1.201565646729436e-05, 1.8065624193210164e-05, 1.8555272964616667e-05, 3.788803054038253e-05, 1.9248189284773046e-05, 1.949872276795654e-05, 5.611819178951454e-05, 1.2621108434092658e-05, 1.4936584623839091e-05, 2.70439019975523e-05, 2.6773195120781602e-05, 3.6168626222674326e-05, 1.992408885543263e-05, 1.7436190475500648e-05, 2.0646101105016332e-05, 1.941491609374857e-05, 1.260487913964565e-05, 1.615513541708446e-05, 4.3090962710698834e-05, 1.671515153042973e-05, 1.893358516356521e-05, 2.3714231530124842e-05, 3.750033944650835e-05, 2.2473367355538482e-05, 7.596346140233492e-05, 2.15650786556081e-05, 1.3308950535556622e-05, 2.537039881493669e-05, 1.542634373019819e-05, 4.642583198292354e-05, 2.4176692308661103e-05, 1.8508303910924482e-05, 2.6071065180551198e-05, 3.5987107576240484e-05, 3.128736707201922e-05, 2.0589382256766746e-05, 2.1027805355321068e-05, 1.6433418359907505e-05, 0.00010026575307775851, 3.525748310271185e-05, 1.88646196089368e-05, 2.4577260392154675e-05, 2.8463162699078373e-05, 2.650083440925291e-05, 3.2617769433088535e-05, 2.1207415986537658e-05, 1.6997150883887615e-05, 2.489254196978257e-05, 2.0430867896478327e-05, 2.105032836992292e-05, 2.1903079180422015e-05, 2.8989320975106804e-05, 1.479848372226034e-05, 1.922730996433362e-05, 7.286935935970933e-05, 1.3313858329417833e-05, 2.196442712880887e-05, 4.980348833325275e-05, 2.8236630294305354e-05, 1.9400736271602718e-05, 2.1874382535291662e-05, 0.00017917930416745266, 2.7842748040422335e-05, 2.1117670364108714e-05, 2.6605767809346895e-05, 1.8112090302074163e-05, 2.786466141147307e-05, 3.8402395721125176e-05, 3.0910935590522556e-05, 0.00015489139691789554, 1.5690946621373113e-05, 1.9654271844415344e-05, 2.3731337650774735e-05, 1.5093605536397942e-05, 2.2218464480071856e-05, 1.7318068291336923e-05, 3.6166946413368816e-05, 1.8369543649875538e-05, 1.827406138877255e-05, 1.6575546907146133e-05, 2.8203162553060385e-05, 2.607645423930021e-05, 1.7614353126748838e-05, 3.076821635238695e-05, 2.758899590544037e-05, 3.082247273911771e-05, 1.5812607952118166e-05, 4.6325732152518545e-05, 3.142576267528526e-05, 3.192539443412117e-05, 1.9443730939907125e-05, 3.1844116079738455e-05, 1.8279064269155738e-05, 2.165829420446895e-05, 2.765501615330777e-05, 3.788475377339273e-05, 2.0043173334728062e-05, 1.8848365245365697e-05, 3.159179377132305e-05, 1.549364877892313e-05, 2.9555310437676387e-05, 1.6320441669112212e-05, 1.8733534078672843e-05, 1.556089714835687e-05, 2.0000894312235146e-05, 8.441150005966277e-05, 3.353374472705145e-05, 2.7820632617463698e-05, 3.147375579891768e-05, 2.988291971028653e-05, 3.3643374173986564e-05, 3.206606940831406e-05, 3.254177771184322e-05, 3.4181845913034515e-05, 2.5528791633822934e-05, 1.7405492402321913e-05, 0.000149769839124403, 3.589954026293597e-05, 2.104549612783794e-05, 1.5759844725927132e-05, 6.134817074900418e-05, 2.8305929831102428e-05, 1.5455072999040495e-05, 6.810969298783467e-05, 0.0001006208266875468, 4.212932824656367e-05, 2.1532759846372972e-05, 2.4969674388580828e-05, 1.7819222079333726e-05, 2.9558444723554485e-05, 2.6073277489591786e-05, 1.8458161410878155e-05, 2.7813887840989478e-05, 1.711336206817045e-05, 1.4634556392636861e-05, 4.575127241825342e-05, 2.9468215812383853e-05, 2.855862499057113e-05, 2.0748165758044706e-05, 2.80523753111111e-05, 2.9057595584744968e-05, 1.2630664565869087e-05, 3.8925501320010506e-05, 2.3441619178236106e-05, 2.748887494619322e-05, 2.4212125396274342e-05, 8.452991722361163e-05, 1.8890199418009145e-05, 2.5004326997143264e-05, 3.64980075143887e-05, 3.960663133265099e-05, 1.4510837203883635e-05, 1.4745410442520771e-05, 3.170536283395625e-05, 4.0007163645269236e-05, 1.868359723649736e-05, 4.5341277427809456e-05, 3.227179420669057e-05, 1.429719047443834e-05, 1.574073274791814e-05, 1.586008634908458e-05, 1.5441233458690256e-05, 2.6454699418767045e-05, 1.6484654647132e-05, 4.6946054242362264e-05, 4.2048023180365086e-05, 3.084369566797541e-05, 1.992260797857783e-05, 2.045261915258011e-05, 4.006099747608495e-05, 2.0003032083585976e-05, 2.0779877871483073e-05, 9.626461167180596e-05, 2.0524598733568362e-05, 4.5180038344572496e-05, 3.90749137380395e-05, 2.2185898382547308e-05, 1.5543423534792917e-05, 5.9324599215910286e-05, 0.00011249517637850383, 4.316338915844506e-05, 2.2658155871044945e-05, 1.899407797214766e-05, 3.9048441267067675e-05, 1.5842918339314466e-05, 2.166609547535581e-05, 2.8940646784741783e-05, 3.0496744297377e-05, 1.7023253809894677e-05, 2.8194138077932404e-05, 2.9562578361381606e-05, 1.5220726045286326e-05, 2.4862107004458164e-05, 4.199838578294709e-05, 2.1693236890773826e-05, 4.7585266266141594e-05, 1.741541996495417e-05, 1.7708982942046832e-05, 2.001358613636422e-05, 4.415849897130512e-05, 3.7410334908930574e-05, 3.0322934296649024e-05, 2.217944328158754e-05, 1.592863304132068e-05, 8.406964718837944e-05, 2.1130793772905684e-05, 4.607992848559837e-05, 2.905861698039749e-05, 1.4079495770662242e-05, 1.6450661773576968e-05, 0.0003069840962354472, 4.2167845156741004e-05, 4.4250647603098946e-05, 1.6898008622878758e-05, 2.075104251378382e-05, 1.9784880874547518e-05, 1.1990244220219043e-05, 4.653497405063505e-05, 3.517244192129937e-05, 1.5791700546978078e-05, 5.5900311324059296e-05, 2.0955772110198306e-05, 2.5728780246947525e-05, 4.18300333812754e-05, 4.513023288849194e-05, 2.4460859353556908e-05, 1.5205723278592355e-05, 4.357463249103264e-05, 2.600466753789183e-05, 4.524206531323392e-05, 2.592921952289781e-05, 4.37462177103443e-05, 1.3387547846529524e-05, 0.0001534426890468662, 7.40006126954141e-05, 1.4717784019335658e-05, 2.1202285285711e-05, 1.8933468472818873e-05, 1.3554032908955432e-05, 1.4036626907375954e-05, 2.5809735218666634e-05, 1.5778439527301924e-05, 2.3393291611112686e-05, 1.525176157175929e-05, 2.8284279564981586e-05, 5.995134961645044e-05, 1.773282659688067e-05, 1.4422334206373612e-05, 1.669687249423981e-05, 1.4792513935302783e-05, 2.2555654185920275e-05, 5.638890794529881e-05, 2.2421258219468307e-05, 4.8769742022957564e-05, 1.7232071833820674e-05, 1.624981565401896e-05, 1.8664679830480917e-05, 1.3365449207290465e-05, 3.574576675745003e-05, 3.208436573815664e-05, 2.5695181685132078e-05, 4.843496534382901e-05, 1.636606445022533e-05, 3.160183937227576e-05, 3.220307003649381e-05, 1.8297691189387756e-05, 4.6785884928110194e-05, 3.613842873947696e-05, 3.120100748670541e-05, 2.8397947403429602e-05, 2.3879229846075498e-05, 1.7642398502511746e-05, 3.796465818556449e-05, 2.258600213344026e-05, 3.6835928720951165e-05, 1.5638035523793107e-05, 2.2954887034328515e-05, 5.101797933085841e-05, 1.7265928408293676e-05, 1.2115708915969657e-05, 1.8990828899904362e-05, 2.5519627493082693e-05, 3.491035843216903e-05, 6.406103935689519e-05, 1.7934375791790316e-05, 3.2060706902191203e-05, 3.8395825629497374e-05, 1.797150669088731e-05, 1.663554935768979e-05, 2.082339891222033e-05, 2.236644278783578e-05, 1.9762583877635393e-05, 2.0144149926672035e-05, 1.0821470517098672e-05, 1.838683806943929e-05, 1.7106358301403567e-05, 3.047010922507334e-05, 2.324108878145731e-05, 2.0915031966059362e-05, 1.9309746524063674e-05, 1.9583186896631677e-05, 2.2448439673955785e-05, 6.228501031107932e-05, 5.38497497073075e-05, 1.898749774657928e-05, 3.0847124778186276e-05, 8.446236818158254e-05, 3.393050139381357e-05, 3.5169362936542183e-05, 3.6384113693802985e-05, 2.754615011383482e-05, 0.00013376817968726605, 2.1033360009420987e-05, 1.4364992684646703e-05, 1.6312428205208978e-05, 3.163403019189705e-05, 2.365460224884213e-05, 1.632401601617423e-05, 1.2095949327862892e-05, 3.1496399784975244e-05, 2.564684190338487e-05, 1.8803274138432576e-05, 1.7742487178480358e-05, 2.337153130515077e-05, 2.0464175389312448e-05, 5.543745188789393e-05, 1.82433213198571e-05, 1.9080997176405556e-05, 5.7329586767602625e-05, 1.854601527846125e-05, 4.4657030001419815e-05, 1.7829512575125578e-05, 3.8268317667348806e-05, 3.0142596695412533e-05, 3.4176413332715784e-05, 1.5427616228741327e-05, 3.874810120971043e-05, 5.839905749760749e-05, 6.693140464219974e-05, 1.3533745200230967e-05, 1.6733184863520813e-05, 3.102647667959996e-05, 2.215801990511277e-05, 1.3753799977139077e-05, 2.803122878168143e-05, 1.71945141642124e-05, 1.4576016945001413e-05, 2.569131216877788e-05, 3.253905716509426e-05, 1.3089417905307096e-05, 2.5896435008930096e-05, 1.3036301973225652e-05, 2.2077120629204985e-05, 2.411589299696999e-05, 2.402343602062029e-05, 2.6142979281116e-05, 3.071324100558474e-05, 5.3660802074048256e-05, 2.28312970505247e-05, 3.8311850796904445e-05, 1.6513125305404503e-05, 5.9911418021280245e-05, 4.7737752662375365e-05, 2.8841200777666266e-05, 2.1512522579574113e-05, 2.4840158073921102e-05, 2.037164212743049e-05, 2.541836620790321e-05, 2.9486249456157596e-05, 2.3613172258287067e-05, 2.675658446575987e-05, 1.5206468259067014e-05, 8.405875041694449e-05, 3.157856057499979e-05, 3.595576207870377e-05, 2.391836678997792e-05, 4.953145905403535e-05, 2.8008148393797216e-05, 2.0566783013227166e-05, 2.748874559902779e-05, 2.7679889963099306e-05, 5.560382747382807e-05, 2.1988795334460007e-05, 4.345420123549822e-05, 4.197691255782965e-05, 2.0295385675778448e-05, 2.546513433839304e-05, 1.4626431911845512e-05, 3.5514478537900216e-05, 3.7659653854306005e-05, 2.237216921449168e-05, 1.832449754449427e-05, 2.0565431468319556e-05, 3.240927986723604e-05, 2.200801176963985e-05, 2.2940099513205243e-05, 3.298494845608842e-05, 4.2457900812857715e-05, 4.554212961812085e-05, 3.6074354853014046e-05, 2.2834684261418308e-05, 3.921920890304892e-05, 8.09994364895675e-05, 1.703442983999826e-05, 1.407560502568997e-05, 2.5426121622218295e-05, 2.5981589941520525e-05, 4.272818791748892e-05, 3.9101993420985485e-05, 2.951200074379483e-05, 1.77419519435404e-05, 1.994332367821078e-05, 3.259789018682899e-05, 2.582839432958873e-05, 1.8661995145697444e-05, 1.9486439748578852e-05, 4.5012486652423805e-05, 3.737983065779547e-05, 2.32410363533993e-05, 1.688646036821162e-05, 1.6779224378416932e-05, 1.9985016346045857e-05, 1.9507859748986124e-05, 4.0465313794658084e-05, 0.00010380912561378035, 5.378418112924627e-05, 2.4518623558707447e-05, 3.113765096395205e-05, 3.135880114082304e-05, 1.8353242451143213e-05, 4.708042429140556e-05, 1.5060276783228838e-05, 4.916567426747579e-05, 1.8816397164333567e-05, 2.1658283262827235e-05, 1.8941319771552245e-05, 3.9400723884547555e-05, 1.6408697983071452e-05, 1.6918811871210054e-05, 0.00015190845373769813, 1.9933979457378484e-05, 1.4629942061397668e-05, 2.4588151477837258e-05, 1.814283926465923e-05, 3.409084177753409e-05, 1.992893523021146e-05, 6.657516619935032e-05, 2.6846935379685122e-05, 6.512251022583843e-05, 2.1972683498065023e-05, 4.514923807535138e-05, 1.52045504974158e-05, 2.4466933778683194e-05, 4.6907670267226294e-05, 5.402265051473748e-05, 2.0026394230003064e-05, 2.4372644422712722e-05, 1.9083186815353792e-05, 5.2052485574488396e-05, 1.6551946968975387e-05, 1.6684582398123514e-05, 1.8148222876511903e-05, 3.7227840076116406e-05, 1.9845402107363046e-05, 2.9852775220729063e-05, 1.9468424545517242e-05, 2.7883912105512352e-05, 1.790172431039085e-05, 6.116712151319879e-05, 1.6342748189379704e-05, 2.1860555745714822e-05, 2.2880129098659945e-05, 2.0477704215826014e-05, 2.2837431271040918e-05, 3.218526675040798e-05, 3.3489602294442446e-05, 2.213607641183298e-05, 1.6391564502000675e-05, 2.602461202760313e-05, 2.474495144674177e-05, 4.4022400612140825e-05, 1.6820104505580484e-05, 2.3148860233600972e-05, 5.2188568585088804e-05, 4.2173904230510046e-05, 1.6453239088575388e-05, 1.7797042655715168e-05, 1.792647093176006e-05, 3.694039863019161e-05, 2.2867057868172632e-05, 1.2938903748368009e-05, 1.2631865137583492e-05, 3.479965500396912e-05, 2.9891307813900205e-05, 2.0587383999796866e-05, 2.1630015942140384e-05, 2.1207954822753858e-05, 7.274200972179618e-05, 3.34808242712106e-05, 0.00012963908027098726, 5.1011288381951914e-05, 1.6945594761771393e-05, 2.683686359602905e-05, 4.6036887595904826e-05, 2.0623502709370007e-05, 2.047489155155594e-05, 3.226918678425015e-05, 2.5210823479752636e-05, 2.4727994639073455e-05, 2.189997590046256e-05, 4.4279511665751036e-05, 3.1360673476784545e-05, 3.201308990934056e-05, 3.5605365334516615e-05, 2.75189926186636e-05, 1.66756968410279e-05, 1.6890592509613434e-05, 2.3432560714225222e-05, 3.0599921376678075e-05, 2.6395587880030294e-05, 2.7671081155473438e-05, 2.229556481609583e-05, 1.9108397464713954e-05, 2.22829580914894e-05, 3.336983329790429e-05, 2.4415367005603445e-05, 5.230450153923068e-05, 2.267472011684867e-05, 8.911253743354248e-05, 3.965819873172048e-05, 3.5294528120645006e-05, 1.2473089298421204e-05, 8.558564069237279e-05, 2.1413588124941962e-05, 1.626356956610961e-05, 2.265843142729852e-05, 2.208066314466989e-05, 2.9551817155959274e-05, 1.5465156157550566e-05, 1.969206037850179e-05, 8.499810553314974e-05, 2.330879419505203e-05, 4.354014376250746e-05, 3.1707487525092365e-05, 1.4223940430927216e-05, 2.6863195864237823e-05, 2.606315695127063e-05, 4.333069802573059e-05, 2.251973049993384e-05, 1.586545740300452e-05, 2.3543501090920908e-05, 3.2047157338042944e-05, 2.925441510502696e-05, 2.287012339507617e-05, 2.0101758131701903e-05, 3.118451168642759e-05, 4.520325139025238e-05, 5.1376043110352105e-05, 1.7193120642409633e-05, 2.071560919199601e-05, 2.7376237056059885e-05, 6.159889318529599e-05, 2.3974527243946823e-05, 2.0357078817407884e-05, 1.6390059916959178e-05, 2.4634177698636697e-05, 1.832985717205483e-05, 3.453223465136825e-05, 1.8984746655442504e-05, 9.813030339964113e-05, 2.0951689484431557e-05, 3.127488669842412e-05, 3.329301419806886e-05, 4.083164282068401e-05, 2.9380203885965714e-05, 7.947032655660332e-05, 1.6088887499993424e-05, 1.529164902008931e-05, 5.0906785268572556e-05, 2.2793890602515855e-05, 1.6321120013042793e-05, 7.267329761160847e-05, 2.238924853775878e-05, 1.746523764969959e-05, 1.9310938454660766e-05, 3.3493567388884736e-05, 1.9728622259574165e-05, 4.759430335318888e-05, 3.842486968287809e-05, 1.6101250089739367e-05, 2.10332101769521e-05, 1.4274831283571283e-05, 2.614685692769556e-05, 3.884227668321421e-05, 2.873077149939819e-05, 9.12088688276573e-05, 2.2375143895032715e-05, 3.766588279288371e-05, 3.6770715647981734e-05, 4.0558389461675157e-05, 1.535054050354451e-05, 2.0016676737763778e-05, 2.1146957269134035e-05, 1.3418327009102992e-05, 1.991341756800566e-05, 2.6359930670078607e-05, 2.318283643572635e-05, 2.3277137005800948e-05, 1.45923662606993e-05, 1.7696755773702886e-05, 1.891724983718303e-05, 2.0014416255812054e-05, 2.491537110082348e-05, 2.918346022555908e-05, 3.4407021807120845e-05, 7.616705771919327e-05, 2.403639022418113e-05, 2.7188576605363582e-05, 5.764094839585013e-05, 2.076363848243221e-05, 2.585839544325381e-05, 1.8757650432927303e-05, 6.56606978558504e-05, 1.2811277260825576e-05]
CI = np.quantile(ls, [0.05, 0.95])  # [1.45064121e-05, 7.12775978e-05]
shap = [0.00823278342451576]

df = pd.DataFrame(ls, columns=["shap_value"])
df["color"] = 1
print(df.head())

# ヒストグラム
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 0.01)
s.distplot(ls)
s.distplot(shap)
plt.show()


# ランダム化前のshap value
shap = pd.DataFrame([[0.00823278342451576, 2]], columns=["shap_value", "color"])
print(shap)
df = pd.concat([df, shap])
df["x"] = 1
print(df)

"""
s.set()
s.set_style('whitegrid')
s.set_palette('Set1')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.hist(ls)  #, alpha=0.6)
ax.hist(shap)  #, alpha=0.6)
ax.set_xlabel('shap value')
ax.set_xlim(0, 0.001)
plt.show()
"""

# swarm plotの表示
s.catplot(x='x', y='shap_value', data=df, kind='swarm', hue='color')
plt.show()

s.catplot(x='x', y='shap_value', data=df, kind='box', hue='color')
plt.show()

"""
s.displot(data=df, x='shap_value', hue='color', multiple='stack')
plt.show()
"""
'''