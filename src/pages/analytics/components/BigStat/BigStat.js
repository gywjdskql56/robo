import React, { Component } from 'react';
import PropTypes from 'prop-types';
import cx from 'classnames';
import {
  UncontrolledButtonDropdown,
  DropdownToggle,
  DropdownMenu,
  DropdownItem,
} from 'reactstrap';

import Widget from '../../../../components/Widget';

import s from './BigStat.module.scss';

class BigStat extends Component {
  static propTypes = {
    product: PropTypes.string.isRequired,
    total: PropTypes.string.isRequired,
    registrations: PropTypes.shape({
      value: PropTypes.number,
      profit: PropTypes.bool,
    }).isRequired,
    bounce: PropTypes.shape({
      value: PropTypes.number,
      profit: PropTypes.bool,
    }).isRequired,
    color: PropTypes.string.isRequired,
  }

  state = { simpleSelectDropdownValue: 'Daily' }

  changeSelectDropdownSimple = (e) => {
    this.setState({ simpleSelectDropdownValue: e.currentTarget.textContent });
  }

  render() {
    const { product, total, registrations, bounce, color } = this.props;
    return (
      <div className="pb-xlg h-100">
        <Widget
          className="mb-0 h-100"
          bodyClass={`mt p-0`}
          title={
            <div className="d-flex justify-content-between flex-wrap">
              <h4 className={cx('d-flex align-items-center pb-1', s.bigStatTitle)}>
                <span className={`circle bg-${color} me-2`} style={{ fontSize: '6px' }} />
                  <span className="fw-normal ms-xs">{product}</span>
              </h4>
              <UncontrolledButtonDropdown>
                <DropdownToggle
                  caret color="default"
                  className="dropdown-toggle-split me-2 btn-sm"
                >
                  {this.state.simpleSelectDropdownValue}&nbsp;&nbsp;
                </DropdownToggle>
                <DropdownMenu>
                  <DropdownItem onClick={this.changeSelectDropdownSimple}>
                    Daily
                  </DropdownItem>
                  <DropdownItem onClick={this.changeSelectDropdownSimple}>
                    Weekly
                  </DropdownItem>
                  <DropdownItem onClick={this.changeSelectDropdownSimple}>
                    Monthly
                  </DropdownItem>
                </DropdownMenu>
              </UncontrolledButtonDropdown>
            </div>
          }
        >
          <h4 className="fw-semi-bold mb-lg px-4">{total}</h4>
          <div className={`d-flex ${s.borderTop}`}>
            <div className={`w-50 ${s.borderRight} py-3 pe-2 ps-4`}>
              <div className="d-flex align-items-start h3">
                <h6>+{registrations.value}</h6>
                <i
                  className={`la la-arrow-right ms-1 text-${registrations.profit ? 'success' : 'danger'}
                  rotate-${registrations.profit ? '315' : '45'}`}
                />
              </div>
              <p className="text-muted mb-0 me-2"><small>Registrations</small></p>
            </div>
            <div className="w-50 py-3 ps-2">
              <div className="d-flex align-items-start h3">
                <h6>{bounce.value}%</h6>
                <i
                  className={`la la-arrow-right ms-sm text-${bounce.profit ? 'success' : 'danger'}
                  rotate-${bounce.profit ? '315' : '45'}`}
                />
              </div>
              <p className="text-muted mb-0 me-2"><small>Bounce Rate</small></p>
            </div>
          </div>
        </Widget>
      </div>
    );
  }
}

export default BigStat;
