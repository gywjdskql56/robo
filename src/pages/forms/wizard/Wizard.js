import React from 'react';
import Rating from '@mui/material/Rating';
import StarIcon from '@mui/icons-material/Star';
import ReactEchartsCore from 'echarts-for-react/lib/core';
import echarts from 'echarts/lib/echarts';
import {chartData, liveChart, liveChartInterval} from './mock';
import chartsDataHigh from './mock_high';
import Highcharts from 'highcharts';
import HighchartsReact from 'highcharts-react-official'
import BootstrapTable from 'react-bootstrap-table-next';
import paginationFactory from 'react-bootstrap-table2-paginator';
import ToolkitProvider, { Search } from 'react-bootstrap-table2-toolkit';
import classnames from 'classnames';
import Typography from '@mui/material/Typography';
import { ResponsiveTreeMap } from '@nivo/treemap'
import moment from 'moment'
import {
  Row,Col,Button,Table,Badge,FormGroup,Label,Input,InputGroup,InputGroupText,Nav,NavLink,NavItem,Progress,
} from 'reactstrap';
import Formsy from 'formsy-react';
import MaskedInput from 'react-maskedinput';
import Datetime from 'react-datetime';
import { select2CountriesData, select2ShipmentData, cardTypesData } from './data';
import Select from 'react-select';
import InputValidation from '../../../components/InputValidation/InputValidation';
import Widget from '../../../components/Widget';
import s from './Wizard.module.scss';
import ss from './Static.module.scss';
import sss from './Elements.module.scss';
import GaugeChart from 'react-gauge-chart'
import { reactTableData, reactBootstrapTableData } from './data_dy';
import {httpGet} from "./config";
import Slider, { Range, createSliderWithTooltip } from 'rc-slider';
const SliderWithTooltip = createSliderWithTooltip(Slider);
const { SearchBar } = Search;
const labels: { [index: string]: string } = {
  1: '안정형(매우낮은위험)',
  2: '안정추구형(낮은위험)',
  3: '위험중립형(보통위험)',
  4: '적극투자형(높은위험)',
  5: '공격투자형(매우높은위험)',
};

const count = 4;



const initEchartsOptions= {
      renderer: 'canvas'
      }
function getLabelText(value: number) {
  return `${value} Star${value !== 1 ? 's' : ''}, ${labels[value]}`;
}
function infoFormatter(cell) {
  return (
    <div>
      <small>
        Type:&nbsp;<span className="fw-semi-bold">{cell.type}</span>
      </small>
      <br />
      <small>
        Dimensions:&nbsp;<span className="fw-semi-bold">{cell.dimensions}</span>
      </small>
    </div>
  );
}

function descriptionFormatter(cell) {
  return (
    <button className="btn-link">
      {cell}
    </button>
  );
}

function progressFormatter(cell) {
  return (
    <Progress style={{ height: '15px' }} color={cell.type} value={cell.progress} />
  );
}

function progressSortFunc(a, b, order) {
  if (order === 'asc') {
    return a.progress - b.progress;
  }
  return b.progress - a.progress;
}

function dateSortFunc(a, b, order) {
  if (order === 'asc') {
    return new Date(a).getTime() - new Date(b).getTime();
  }
  return new Date(b).getTime() - new Date(a).getTime();
}

function sortableHeaderFormatter(column, index, components) {
  let icon = (<div className={classnames(s.sortArrowsContainer, 'ms-sm')}>
    <span className={classnames('caret', 'mb-xs', s.rotatedArrow)} />
    <span className="caret"/>
  </div>);

  if (components.sortElement.props.order === 'asc') {
    icon = (<div className="ms-sm">
      <span className={classnames('caret', 'mb-xs', s.rotatedArrow)}/>
    </div>);
  } else if (components.sortElement.props.order === 'desc') {
    icon = (<div className="ms-sm">
      <span className="caret mb-xs"/>
    </div>);
  }

  return <div className={s.sortableHeaderContainer}>{column.text} {icon}</div>
}
//const checkboxes1 = [false, false, true, false];
//sessionStorage.setItem('checkboxes', JSON.stringify(checkboxes1));

const StepsComponents = {
  Step1: function Step1() {
    const [value, setValue] = React.useState(3);
    const [legend, setLegend] = React.useState("중수익 중위험을 선호합니다.");
    const [input1, setInput1] = React.useState(0);
    const [input2, setInput2] = React.useState(0);
    const [input3, setInput3] = React.useState("");
    const handleChange1 = (e) => {
        setInput1(e.target.value);
        sessionStorage.setItem("input1", e.target.value);
    };
    const handleChange3 = (e) => {
        setInput3(e.target.value);
        sessionStorage.setItem("input3", e.target.value);
    };
    return (<FormGroup row>
    <Col xs={4} xl={4} md={4}>
    <Label md={4} className="text-md-right" for="appended-input">
      투자 성향
    </Label>
      <Typography component="legend">{legend}</Typography>
      <Rating
        name="hover-feedback"
        value={value}
        precision={1}
        onChange={(event, newValue) => {
          setValue(newValue);
          sessionStorage.setItem("value", newValue);
          console.log(sessionStorage.getItem("value"))
          if (newValue == 1) {
            setLegend("매우 보수적인 성향으로 최대한 적은 위험을 선호합니다.")
          } else if (newValue == 2) {
            setLegend("보수적인 성향으로 적은 위험을 선호합니다.")
          } else if (newValue == 3) {
            setLegend("중수익 중위험을 선호합니다.")
          } else if (newValue == 4) {
            setLegend("높은 위험을 감수할 수 있습니다.")
          } else if (newValue == 5) {
            setLegend("매우 공격적인 성향으로, 높은 위험을 감수할 수 있습니다.")
          }

        }}
        getLabelText={getLabelText}
        emptyIcon={<StarIcon style={{ opacity: 0.55 }} fontSize="inherit" />}
      />
      </Col>
      <Col xs={3} xl={3} md={3}>
            <Label md={4} className="text-md-right" for="appended-input">
              목표 금액
            </Label>
              <InputGroup>
                <Input className="input-transparent" id="appended-input" bsSize="16" type="text" onChange={handleChange1} />
                <InputGroupText>만원</InputGroupText>
              </InputGroup>
          </Col>
          <Col xs={3} xl={3} md={3}>
          <Label md={4} className="text-md-right" for="appended-input">투자 가능기간</Label>
              <InputGroup>
                <Input className="input-transparent" id="appended-input" bsSize="16" type="text" onChange={handleChange3} />
                <InputGroupText>년</InputGroupText>
              </InputGroup>
              {/*<div className="datepicker" style={{display: 'flex'}}>
                <Datetime
                  id="datepicker"
                  viewMode="days"
                  timeFormat={false}
                  onChange={(date)=>{
                    setInput3(moment(date).year());
                    sessionStorage.setItem("input3", moment(date).year())
                  }}
                />
                <InputGroupText>
                  <i className="glyphicon glyphicon-th" />
                </InputGroupText>
              </div>*/}
          </Col>
    </FormGroup>);
  },
  Step2: function Step2() {
      const [input, setInput] = React.useState(0);
      const [min, setMin] = React.useState(0);
      const [max, setMax] = React.useState(100);
      const [gauge, setGauge] = React.useState(0);

      console.log(sessionStorage.getItem("get_range_max"))
      console.log(typeof(sessionStorage.getItem("get_range_max")))
      console.log(sessionStorage.getItem("get_range_min"))
    return (
      <fieldset>
        <FormGroup row>
          <Col xs={6} xl={6} md={6}>
            <Label md={4} className="text-md-right" for="appended-input">
              초기투자금액
            </Label>
              <div className="slider-warning mb-sm">
                <SliderWithTooltip
                  tipFormatter={(v) => {
                    return `${v} 만원`;
                  }}
                  className={`${sss.sliderCustomization} ${sss.horizontalSlider} ${sss.sliderYellow}`}
                  defaultValue={sessionStorage.getItem("get_range_min")}
                  min={parseInt(sessionStorage.getItem("get_range_min"))}//{sessionStorage.getItem("get_range_min")}
                  max={parseInt(sessionStorage.getItem("get_range_max"))}//{sessionStorage.getItem("get_range_max")}
                  onChange={(v) => {
                    console.log(v)
                    setGauge(Math.max(v/parseInt(sessionStorage.getItem("get_range_max")))-0.09,0)
                    sessionStorage.setItem("init_invest", v)
                    }}

                />
              </div>
              {/*<InputGroup>
                <Input className="input-transparent" id="appended-input" bsSize="16" type="text" onChange={handleChange2}/>
                <InputGroupText>만원</InputGroupText>
              </InputGroup>*/}
          </Col>
          <Col xs={6} xl={6} md={6}>
            <Label for="datetimepicker">목표달성 확률</Label>
            <GaugeChart
                id="gauge-chart1"
                style={{height: 250}}
                colors={["#FF0000", "#008000"]}
                nrOfLevels={4}
                formatTextValue={(value) => value+"%"}
                percent={gauge}
             />
          </Col>
        </FormGroup>
        {/*<FormGroup>
          <Label for="country-select">Destination Country</Label>
            <Select
              classNamePrefix="react-select"
              className="selectCustomization"
              options={select2CountriesData}
            />
          <p className="help-block">Please choose your country destination</p>
        </FormGroup>
        <FormGroup>
          <Label for="courier">Choose shipping option</Label>
            <Select
              classNamePrefix="react-select"
              className="selectCustomization"
              options={select2ShipmentData}
            />
          <p className="help-block">Please choose your shipping option</p>
        </FormGroup>
        <FormGroup>
          <Label for="destination">Destination Zip Code</Label>
          <MaskedInput
            className="form-control" id="destination" mask="111111"
            size="6"
          />
          <p className="help-block">Please provide your Destination Zip Code</p>
        </FormGroup>
        <FormGroup>
          <Label for="dest-address">Destination Address</Label>
          <InputValidation type="text" id="dest-address" name="dest-address" />
          <p className="help-block">Please provide the destination address</p>
        </FormGroup>*/}
      </fieldset>
    );
  },
  Step3: function Step3(props) {
    const [checkboxes1, SetCheckboxes1] = React.useState([false, false, false, true]);
    const [risktype, SetRisktype] = React.useState("Conservative");
    sessionStorage.setItem("risktype","Conservative")
    return (
      <fieldset>
              <div className={`widget-table-overflow ${ss.overFlow}`}>
                <Table className="table-striped">
                  <thead>
                    <tr>
                      <th>
                      </th>
                      <th>위험유형</th>
                      <th>목표달성률</th>
                      <th>원금손실률</th>
                      <th>기대수익률</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                      </td>
                      <td><span className="badge bg-gray">60:40포트폴리오</span></td>
                      <td>-%</td>
                      <td>-%</td>
                      <td>-%</td>
                    </tr>
                    <tr>
                      <td>
                        <div className="abc-checkbox">
                          <Input
                            id="checkbox2" type="checkbox" checked={checkboxes1[1]}
                            onClick={() => {SetCheckboxes1([false, true, false, false]); SetRisktype("Aggressive"); sessionStorage.setItem("risktype","Aggressive")} }
                          />
                          <Label for="checkbox2" />
                        </div>
                      </td>
                      <td><Badge color="danger">공격형</Badge></td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Aggressive.goal_prob}%</td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Aggressive.loss_prob}%</td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Aggressive["return"]}%</td>
                    </tr>
                    <tr>
                      <td>
                        <div className="abc-checkbox">
                          <Input
                            id="checkbox3" type="checkbox" checked={checkboxes1[2]}
                            onClick={() => {SetCheckboxes1([false, false, true, false]); SetRisktype("Neutral");sessionStorage.setItem("risktype","Neutral")} }
                          />
                          <Label for="checkbox3" />
                        </div>
                      </td>
                      <td><Badge color="warning" className="text-gray-dark">위험중립형</Badge></td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Neutral.goal_prob}%</td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Neutral.loss_prob}%</td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Neutral["return"]}%</td>
                    </tr>
                    <tr>
                      <td>
                        <div className="abc-checkbox">
                          <Input
                            id="checkbox4" type="checkbox" checked={checkboxes1[3]}
                            onClick={() => {SetCheckboxes1([false, false, false, true]); SetRisktype("Conservative");sessionStorage.setItem("risktype","Conservative")} }
                          />
                          <Label for="checkbox4" />
                        </div>
                      </td>
                      <td><Badge color="success">안전추구형</Badge></td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Conservative.goal_prob}%<Badge color="info" className="text-gray-dark">제안</Badge></td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Conservative.loss_prob}%</td>
                      <td>{JSON.parse(sessionStorage.getItem("init_port")).Conservative["return"]}%</td>
                    </tr>
                  </tbody>
                </Table>
              </div>
              <ReactEchartsCore
               echarts={echarts}
               option={chartData.echarts.line}
               opts={initEchartsOptions}
               style={{height: "365px"}}
              />
              <ReactEchartsCore
                echarts={echarts}
                option={chartData.echarts.donut}
                opts={initEchartsOptions}
                style={{height: "170px"}}
              />
              <ReactEchartsCore
                  echarts={echarts}
                  option={chartData.echarts.river}
                  opts={initEchartsOptions}
                  style={{height: "350px"}}
              />

        {/*<FormGroup>
          <Label for="name">Name on the Card</Label>
          <InputValidation type="text" id="name" name="name" />
        </FormGroup>
        <FormGroup>
          <Label for="credit-card-type">Choose shipping option</Label>
            <Select
              classNamePrefix="react-select"
              className="selectCustomization"
              options={cardTypesData}
            />
        </FormGroup>
        <FormGroup>
          <Label for="credit">Credit Card Number</Label>
          <InputValidation type="text" id="credit" name="credit" />
        </FormGroup>
        <FormGroup>
          <Label for="expiration-data">Expiration Date</Label>
          <div className="datepicker">
            <Datetime
              id="datepicker"
              open={props.isDatePickerOpen} //eslint-disable-line
              viewMode="days"
            />
          </div>
        </FormGroup>*/}
      </fieldset>
    );
  },
  Step4: function Step4() {
    return (
      <fieldset>

        <h2>포트폴리오 생성이 완료되었습니다.</h2>
        <p>분석화면을 통해 생성된 포트폴리오의 자세한 정보를 확인해보세요.</p>
          <Col lg={12} xs={12}>
            <Widget
              title={<h5>초기 투자 <span className="fw-semi-bold">포트폴리오 </span> (초기 투자액 : {sessionStorage.getItem("init_invest")}만원)</h5>}
              close collapse
            >
              <HighchartsReact highcharts={Highcharts} options={chartsDataHigh.pie} />
            </Widget>
          </Col>
          <Widget className="table-responsive" title={<h4><span className="fw-semi-bold">포트폴리오</span> 보유내역</h4>} collapse close>
          <ToolkitProvider
            keyField="id"
            data={reactBootstrapTableData()}
            columns={[{
              dataField: 'id',
              text: 'ID'
            }, {
              dataField: 'name',
              text: 'Name'
            }, {
              dataField: 'description',
              text: '종목 이름',
              formatter: descriptionFormatter,
            }, {
              dataField: 'info',
              text: '투자금액',
              formatter: descriptionFormatter,
            }, {
              dataField: 'date',
              text: '투자비중',
              formatter: progressFormatter,
              sort: true,
              sortFunc: progressSortFunc,
              headerFormatter: sortableHeaderFormatter,
            }, {
              dataField: 'status',
              text: '한달간 수익률',
              formatter: progressFormatter,
              sort: true,
              sortFunc: progressSortFunc,
              headerFormatter: sortableHeaderFormatter,
            }]}
            search
          >
            {
              props => (
                <div>
                  <Row className="mb-lg float-end">
                    <Col>
                      <SearchBar { ...props.searchProps } />
                    </Col>
                  </Row>
                  <BootstrapTable
                    { ...props.baseProps }
                    pagination={paginationFactory()}
                    striped
                  />
                </div>
              )
            }
          </ToolkitProvider>
        </Widget>
      </fieldset>
    );
  },

};

class Wizard extends React.Component {
  constructor(prop) {
    super(prop);
    this.state = {
      currentStep: 1,
      progress: 25,
      isDatePickerOpen: false,
      starvalue:3,
      hover:-1,
      label:'',
    };
    this.nextStep = this.nextStep.bind(this);
    this.previousStep = this.previousStep.bind(this);
  }

   getLabelText(value: number) {
    this.setState({
      label:`${value} Star${value !== 1 ? 's' : ''}, ${labels[value]}`,
    });
  }


  nextStep() {
    let currentStep = this.state.currentStep;
    if (currentStep >= count) {
      currentStep = count;
    } else {
      currentStep += 1;
    }

    this.setState({
      currentStep,
      progress: (100 / count) * currentStep,
    });
    if (currentStep==2) {
        const range = httpGet('/get_init_range/'+sessionStorage.getItem("value")+'_'+sessionStorage.getItem("input1")+'_'+sessionStorage.getItem("input3"))
        sessionStorage.setItem("get_range_min", parseInt(range.min))
        sessionStorage.setItem("get_range_max", parseInt(range.max))
        console.log(sessionStorage.getItem("value"))
//      httpGet('/make_port_by_type/'+sessionStorage.getItem("value"))
    }
    if (currentStep==3) {
        const port = httpGet('/get_init_port/'+sessionStorage.getItem("value")+'_'+sessionStorage.getItem("input1")+'_'+sessionStorage.getItem("input3")+'_'+sessionStorage.getItem("init_invest"))
        sessionStorage.setItem("init_port", JSON.stringify(port))
        console.log(JSON.parse(sessionStorage.getItem("init_port")).Conservative.sec_weight_df)
        console.log(JSON.parse(sessionStorage.getItem("init_port")).Conservative.sec_weight_tickers)
        console.log(JSON.parse(sessionStorage.getItem("init_port")).Conservative.init_port)
        chartData.echarts.line.xAxis[0].data = JSON.parse(sessionStorage.getItem("init_port")).Conservative.wealth_path_idx// 기간별 목표수익률 wealth_path_idx
        chartData.echarts.line.xAxis[1].data = JSON.parse(sessionStorage.getItem("init_port")).Conservative.wealth_path_idx// 예상 자산 wealth_path_idx
        chartData.echarts.line.series[0].data = JSON.parse(sessionStorage.getItem("init_port")).Conservative.wealth_path_goal// 목표 자산 wealth_path_goal
        chartData.echarts.line.series[1].data = JSON.parse(sessionStorage.getItem("init_port")).Conservative.wealth_path// 예상 자산 wealth_path
        chartData.echarts.line.series[2].data = JSON.parse(sessionStorage.getItem("init_port")).Conservative.wealth_path_goal_step// 기간별 목표 자산 wealth_path_goal_step
    }
  }

  previousStep() {
    let currentStep = this.state.currentStep;
    if (currentStep <= 1) {
      currentStep = 1;
    } else {
      currentStep -= 1;
    }

    this.setState({
      currentStep,
      progress: (100 / count) * currentStep,
    });
  }

  render() {
    const currentStep = this.state.currentStep;
    return (
      <div className={s.root}>
        <h1 className="page-title"><span className="fw-semi-bold">포트폴리오 </span>생성
        </h1>
        <Row>
          <Col sm={12}>
            <Widget
              close collapse
              className={s.formWizard}
              title={<div>
                <h4>
                  GBI 기반 알고리즘을 활용하여
                  <small> 투자자의 성향, 목표, 투자기간</small>
                </h4>
                <p className="text-muted">에 꼭 맞는 투자 포트폴리오를 제시합니다.</p></div>}
            >

              <Nav pills justified className={s.wizardNavigation}>
                <NavItem>
                  <NavLink active={currentStep === 1}>
                    <small>1.</small>
                    &nbsp;
                    투자자 성향
                  </NavLink>
                </NavItem>
                <NavItem>
                  <NavLink active={currentStep === 2}>
                    <small>2.</small>
                    &nbsp;
                    투자자의 목표수준 및 현재상황
                  </NavLink>
                </NavItem>
                <NavItem>
                  <NavLink active={currentStep === 3}>
                    <small>3.</small>
                    &nbsp;
                    포트폴리오 선택
                  </NavLink>
                </NavItem>
                <NavItem>
                  <NavLink active={currentStep === 4}>
                    <small>4.</small>
                    &nbsp;
                    포트폴리오 생성
                  </NavLink>
                </NavItem>
              </Nav>
              <Progress value={this.state.progress} color="success" className="progress-xs" />
              <div className="tab-content">
                <div className={s.stepBody}>
                  <Formsy.Form>
                    {currentStep === 1 && <StepsComponents.Step1 />}
                    {currentStep === 2 && <StepsComponents.Step2 />}
                    {currentStep === 3 && <StepsComponents.Step3 />}
                    {currentStep === 4 &&
                    <StepsComponents.Step4 isDatePickerOpen={this.state.isDatePickerOpen} />}
                  </Formsy.Form>
                </div>

                <div className="description">
                  <ul className="pager wizard">
                    <li className="previous">
                      <Button hidden={currentStep === 1} color="primary" onClick={this.previousStep}><i
                        className="fa fa-caret-left"
                      />
                        &nbsp;Previous</Button>
                    </li>
                    {/* {currentStep > 1 
                      ? <li className="previous">
                          <Button color="primary" onClick={this.previousStep}><i 
                            className="fa fa-caret-left"
                        />
                          &nbsp;Previous</Button>
                        </li>
                      : <li className="previous">
                          <Button hidden="true" color="primary" onClick={this.previousStep}><i 
                           className="fa fa-caret-left"
                        />
                          &nbsp;Previous</Button>
                        </li>
                    } */}
                    {currentStep < count &&
                    <li className="next">
                      <Button color="primary" onClick={this.nextStep}>Next <i className="fa fa-caret-right" /></Button>
                    </li>
                    }
                    {currentStep === count &&
                    <li className="finish">
                      <Button color="success">Finish <i className="fa fa-check" /></Button>
                    </li>
                    }
                  </ul>
                </div>
              </div>
            </Widget>
          </Col>
        </Row>
      </div>
    );
  }
}

export default Wizard;
