/*
 *  This file is part of RawTherapee.
 *
 *  Copyright (c) 2004-2010 Gabor Horvath <hgabor@rawtherapee.com>
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "xtransprocess.h"

#include "eventmapper.h"
#include "guiutils.h"
#include "options.h"

#include "rtengine/procparams.h"

using namespace rtengine;
using namespace rtengine::procparams;

const Glib::ustring XTransProcess::TOOL_NAME = "xtransprocess";

XTransProcess::XTransProcess () : FoldableToolPanel(this, TOOL_NAME, M("TP_RAW_LABEL"), App::get().options().prevdemo != PD_Sidecar)
{
    auto m = ProcEventMapper::getInstance();
    EvDemosaicBorder = m->newEvent(DEMOSAIC, "HISTORY_MSG_RAW_BORDER");
    EvDemosaicContrast = m->newEvent(DEMOSAIC, "HISTORY_MSG_DUALDEMOSAIC_CONTRAST");
    EvDemosaicAutoContrast = m->newEvent(DEMOSAIC, "HISTORY_MSG_DUALDEMOSAIC_AUTO_CONTRAST");
    EvRafinazariSettings = m->newEvent(DEMOSAIC, "HISTORY_MSG_RAFINAZARI_SETTINGS");

    Gtk::Box* hb1 = Gtk::manage (new Gtk::Box ());
    hb1->pack_start (*Gtk::manage (new Gtk::Label ( M("TP_RAW_DMETHOD") + ": ")), Gtk::PACK_SHRINK, 4);
    method = Gtk::manage (new MyComboBoxText ());

    for (const auto method_string : RAWParams::XTransSensor::getMethodStrings()) {
        const std::string langKey =
            [method_string]() -> std::string
            {
                const std::string str(method_string);

                std::string res;
                for (const auto& c : str) {
                    switch (c) {
                        case '(':
                        case ')':
                        case ' ':
                        case '-': {
                            continue;
                        }

                        default: {
                            res += c;
                            break;
                        }
                    }
                }

                return res;
            }();
        method->append(M("TP_RAW_" + Glib::ustring(langKey).uppercase()));
    }

    method->set_active(0);
    hb1->set_tooltip_markup (M("TP_RAW_SENSOR_XTRANS_DMETHOD_TOOLTIP"));

    hb1->pack_end (*method, Gtk::PACK_EXPAND_WIDGET, 4);
    pack_start( *hb1, Gtk::PACK_SHRINK, 4);

    const auto& options = App::get().options();
    dualDemosaicOptions = Gtk::manage (new Gtk::Box(Gtk::ORIENTATION_VERTICAL));

    dualDemosaicContrast = Gtk::manage(new Adjuster (M("TP_RAW_DUALDEMOSAICCONTRAST"), 0, 100, 1, 20));
    dualDemosaicContrast->setAdjusterListener (this);
    dualDemosaicContrast->addAutoButton(M("TP_RAW_DUALDEMOSAICAUTOCONTRAST_TOOLTIP"));
    dualDemosaicContrast->setAutoValue(true);

    dualDemosaicContrast->setDelay(std::max(options.adjusterMinDelay, options.adjusterMaxDelay));

    dualDemosaicContrast->show();
    dualDemosaicOptions->pack_start(*dualDemosaicContrast);
    pack_start( *dualDemosaicOptions, Gtk::PACK_SHRINK, 4);

    rafinazariOptions = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_VERTICAL));
    rafinazariSigma = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_SIGMA"), 0.50, 5.00, 0.01, 2.32));
    rafinazariNearRadius = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_NEAR_RADIUS"), 1, 24, 1, 7));
    rafinazariMiddleRadius = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_MIDDLE_RADIUS"), 1, 24, 1, 7));
    rafinazariFarRadius = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_FAR_RADIUS"), 1, 24, 1, 7));
    rafinazariEnergySigma = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_ENERGY_SIGMA"), 0.50, 5.00, 0.01, 2.32));
    rafinazariEnergyRadius = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_ENERGY_RADIUS"), 1, 24, 1, 7));
    rafinazariEnergyBoxRadius = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_ENERGY_BOX_RADIUS"), 0, 12, 1, 2));
    rafinazariEnergyFloor = Gtk::manage(new Adjuster(M("TP_RAW_RAFINAZARI_ENERGY_FLOOR"), 0.0, 10.0, 0.001, 0.0));

    for (Adjuster* adjuster : {rafinazariSigma, rafinazariNearRadius, rafinazariMiddleRadius,
                               rafinazariFarRadius, rafinazariEnergySigma, rafinazariEnergyRadius,
                               rafinazariEnergyBoxRadius, rafinazariEnergyFloor}) {
        adjuster->setAdjusterListener(this);
        adjuster->setDelay(std::max(options.adjusterMinDelay, options.adjusterMaxDelay));
        adjuster->show();
        rafinazariOptions->pack_start(*adjuster);
    }
    rafinazariOptions->set_tooltip_markup(M("TP_RAW_RAFINAZARI_TOOLTIP"));
    pack_start(*rafinazariOptions, Gtk::PACK_SHRINK, 4);

    borderbox = Gtk::manage(new Gtk::Box());
    border = Gtk::manage(new Adjuster(M("TP_RAW_BORDER"), 0, 16, 1, 7));
    border->setAdjusterListener (this);

    border->setDelay(std::max(options.adjusterMinDelay, options.adjusterMaxDelay));

    border->show();
    borderbox->pack_start(*border);
    pack_start(*borderbox, Gtk::PACK_SHRINK, 4);

    pack_start( *Gtk::manage( new Gtk::Separator(Gtk::ORIENTATION_HORIZONTAL)), Gtk::PACK_SHRINK, 0 );
    ccSteps = Gtk::manage (new Adjuster (M("TP_RAW_FALSECOLOR"), 0, 5, 1, 0 ));
    ccSteps->setAdjusterListener (this);

    ccSteps->setDelay(std::max(options.adjusterMinDelay, options.adjusterMaxDelay));

    ccSteps->show();
    pack_start( *ccSteps, Gtk::PACK_SHRINK, 4);

    methodconn = method->signal_changed().connect( sigc::mem_fun(*this, &XTransProcess::methodChanged) );
}

XTransProcess::~XTransProcess () {
    idle_register.destroy();
}

void XTransProcess::read(const rtengine::procparams::ProcParams* pp, const ParamsEdited* pedited)
{
    disableListener ();
    methodconn.block (true);

    border->setValue(pp->raw.xtranssensor.border);
    rafinazariSigma->setValue(pp->raw.xtranssensor.rafinazariSigma);
    rafinazariNearRadius->setValue(pp->raw.xtranssensor.rafinazariNearRadius);
    rafinazariMiddleRadius->setValue(pp->raw.xtranssensor.rafinazariMiddleRadius);
    rafinazariFarRadius->setValue(pp->raw.xtranssensor.rafinazariFarRadius);
    rafinazariEnergySigma->setValue(pp->raw.xtranssensor.rafinazariEnergySigma);
    rafinazariEnergyRadius->setValue(pp->raw.xtranssensor.rafinazariEnergyRadius);
    rafinazariEnergyBoxRadius->setValue(pp->raw.xtranssensor.rafinazariEnergyBoxRadius);
    rafinazariEnergyFloor->setValue(pp->raw.xtranssensor.rafinazariEnergyFloor);
    for (size_t i = 0; i < RAWParams::XTransSensor::getMethodStrings().size(); ++i)
        if( pp->raw.xtranssensor.method == RAWParams::XTransSensor::getMethodStrings()[i]) {
            method->set_active(i);
            oldSelection = i;
            break;
        }

    if(pedited ) {
        border->setEditedState (pedited->raw.xtranssensor.border ? Edited : UnEdited);
        dualDemosaicContrast->setAutoInconsistent   (multiImage && !pedited->raw.xtranssensor.dualDemosaicAutoContrast);
        dualDemosaicContrast->setEditedState ( pedited->raw.xtranssensor.dualDemosaicContrast ? Edited : UnEdited);
        ccSteps->setEditedState (pedited->raw.xtranssensor.ccSteps ? Edited : UnEdited);
        rafinazariSigma->setEditedState(pedited->raw.xtranssensor.rafinazariSigma ? Edited : UnEdited);
        rafinazariNearRadius->setEditedState(pedited->raw.xtranssensor.rafinazariNearRadius ? Edited : UnEdited);
        rafinazariMiddleRadius->setEditedState(pedited->raw.xtranssensor.rafinazariMiddleRadius ? Edited : UnEdited);
        rafinazariFarRadius->setEditedState(pedited->raw.xtranssensor.rafinazariFarRadius ? Edited : UnEdited);
        rafinazariEnergySigma->setEditedState(pedited->raw.xtranssensor.rafinazariEnergySigma ? Edited : UnEdited);
        rafinazariEnergyRadius->setEditedState(pedited->raw.xtranssensor.rafinazariEnergyRadius ? Edited : UnEdited);
        rafinazariEnergyBoxRadius->setEditedState(pedited->raw.xtranssensor.rafinazariEnergyBoxRadius ? Edited : UnEdited);
        rafinazariEnergyFloor->setEditedState(pedited->raw.xtranssensor.rafinazariEnergyFloor ? Edited : UnEdited);

        if( !pedited->raw.xtranssensor.method ) {
            method->set_active_text(M("GENERAL_UNCHANGED"));
        }
    }
    dualDemosaicContrast->setAutoValue(pp->raw.xtranssensor.dualDemosaicAutoContrast);
    dualDemosaicContrast->setValue (pp->raw.xtranssensor.dualDemosaicContrast);
    ccSteps->setValue (pp->raw.xtranssensor.ccSteps);

    lastAutoContrast = pp->raw.xtranssensor.dualDemosaicAutoContrast;

    if (!batchMode) {
        dualDemosaicOptions->set_visible(pp->raw.xtranssensor.method == procparams::RAWParams::XTransSensor::getMethodString(procparams::RAWParams::XTransSensor::Method::FOUR_PASS)
                                         || pp->raw.xtranssensor.method == procparams::RAWParams::XTransSensor::getMethodString(procparams::RAWParams::XTransSensor::Method::TWO_PASS));
        rafinazariOptions->set_visible(pp->raw.xtranssensor.method == procparams::RAWParams::XTransSensor::getMethodString(procparams::RAWParams::XTransSensor::Method::RAFINAZARI));
    }

    methodconn.block (false);

    enableListener ();
}

void XTransProcess::write( rtengine::procparams::ProcParams* pp, ParamsEdited* pedited)
{
    pp->raw.xtranssensor.dualDemosaicAutoContrast = dualDemosaicContrast->getAutoValue();
    pp->raw.xtranssensor.dualDemosaicContrast = dualDemosaicContrast->getValue();
    pp->raw.xtranssensor.border = border->getIntValue();
    pp->raw.xtranssensor.ccSteps = ccSteps->getIntValue();
    pp->raw.xtranssensor.rafinazariSigma = rafinazariSigma->getValue();
    pp->raw.xtranssensor.rafinazariNearRadius = rafinazariNearRadius->getIntValue();
    pp->raw.xtranssensor.rafinazariMiddleRadius = rafinazariMiddleRadius->getIntValue();
    pp->raw.xtranssensor.rafinazariFarRadius = rafinazariFarRadius->getIntValue();
    pp->raw.xtranssensor.rafinazariEnergySigma = rafinazariEnergySigma->getValue();
    pp->raw.xtranssensor.rafinazariEnergyRadius = rafinazariEnergyRadius->getIntValue();
    pp->raw.xtranssensor.rafinazariEnergyBoxRadius = rafinazariEnergyBoxRadius->getIntValue();
    pp->raw.xtranssensor.rafinazariEnergyFloor = rafinazariEnergyFloor->getValue();

    int currentRow = method->get_active_row_number();

    if (currentRow >= 0 && method->get_active_text() != M("GENERAL_UNCHANGED")) {
        pp->raw.xtranssensor.method = procparams::RAWParams::XTransSensor::getMethodStrings()[currentRow];
    }

    if (pedited) {
        pedited->raw.xtranssensor.border = border->getEditedState ();
        pedited->raw.xtranssensor.method = method->get_active_text() != M("GENERAL_UNCHANGED");
        pedited->raw.xtranssensor.dualDemosaicAutoContrast = !dualDemosaicContrast->getAutoInconsistent ();
        pedited->raw.xtranssensor.dualDemosaicContrast = dualDemosaicContrast->getEditedState ();
        pedited->raw.xtranssensor.ccSteps = ccSteps->getEditedState ();
        pedited->raw.xtranssensor.rafinazariSigma = rafinazariSigma->getEditedState();
        pedited->raw.xtranssensor.rafinazariNearRadius = rafinazariNearRadius->getEditedState();
        pedited->raw.xtranssensor.rafinazariMiddleRadius = rafinazariMiddleRadius->getEditedState();
        pedited->raw.xtranssensor.rafinazariFarRadius = rafinazariFarRadius->getEditedState();
        pedited->raw.xtranssensor.rafinazariEnergySigma = rafinazariEnergySigma->getEditedState();
        pedited->raw.xtranssensor.rafinazariEnergyRadius = rafinazariEnergyRadius->getEditedState();
        pedited->raw.xtranssensor.rafinazariEnergyBoxRadius = rafinazariEnergyBoxRadius->getEditedState();
        pedited->raw.xtranssensor.rafinazariEnergyFloor = rafinazariEnergyFloor->getEditedState();
    }
}

void XTransProcess::setAdjusterBehavior (bool falsecoloradd, bool dualDemosaicContrastAdd)
{
    border->setAddMode(false);
    dualDemosaicContrast->setAddMode(dualDemosaicContrastAdd);
    ccSteps->setAddMode(falsecoloradd);
    rafinazariSigma->setAddMode(false);
    rafinazariNearRadius->setAddMode(false);
    rafinazariMiddleRadius->setAddMode(false);
    rafinazariFarRadius->setAddMode(false);
    rafinazariEnergySigma->setAddMode(false);
    rafinazariEnergyRadius->setAddMode(false);
    rafinazariEnergyBoxRadius->setAddMode(false);
    rafinazariEnergyFloor->setAddMode(false);
}

void XTransProcess::setBatchMode(bool batchMode)
{
    method->append (M("GENERAL_UNCHANGED"));
    method->set_active_text(M("GENERAL_UNCHANGED"));
    ToolPanel::setBatchMode (batchMode);
    dualDemosaicContrast->showEditedCB ();
    border->showEditedCB ();
    ccSteps->showEditedCB ();
    rafinazariSigma->showEditedCB();
    rafinazariNearRadius->showEditedCB();
    rafinazariMiddleRadius->showEditedCB();
    rafinazariFarRadius->showEditedCB();
    rafinazariEnergySigma->showEditedCB();
    rafinazariEnergyRadius->showEditedCB();
    rafinazariEnergyBoxRadius->showEditedCB();
    rafinazariEnergyFloor->showEditedCB();
}

void XTransProcess::setDefaults(const rtengine::procparams::ProcParams* defParams, const ParamsEdited* pedited)
{
    dualDemosaicContrast->setDefault( defParams->raw.xtranssensor.dualDemosaicContrast);
    border->setDefault (defParams->raw.xtranssensor.border);
    ccSteps->setDefault (defParams->raw.xtranssensor.ccSteps);
    rafinazariSigma->setDefault(defParams->raw.xtranssensor.rafinazariSigma);
    rafinazariNearRadius->setDefault(defParams->raw.xtranssensor.rafinazariNearRadius);
    rafinazariMiddleRadius->setDefault(defParams->raw.xtranssensor.rafinazariMiddleRadius);
    rafinazariFarRadius->setDefault(defParams->raw.xtranssensor.rafinazariFarRadius);
    rafinazariEnergySigma->setDefault(defParams->raw.xtranssensor.rafinazariEnergySigma);
    rafinazariEnergyRadius->setDefault(defParams->raw.xtranssensor.rafinazariEnergyRadius);
    rafinazariEnergyBoxRadius->setDefault(defParams->raw.xtranssensor.rafinazariEnergyBoxRadius);
    rafinazariEnergyFloor->setDefault(defParams->raw.xtranssensor.rafinazariEnergyFloor);

    if (pedited) {
        dualDemosaicContrast->setDefaultEditedState( pedited->raw.xtranssensor.dualDemosaicContrast ? Edited : UnEdited);
        border->setDefaultEditedState(pedited->raw.xtranssensor.border ? Edited : UnEdited);
        ccSteps->setDefaultEditedState(pedited->raw.xtranssensor.ccSteps ? Edited : UnEdited);
        rafinazariSigma->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariSigma ? Edited : UnEdited);
        rafinazariNearRadius->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariNearRadius ? Edited : UnEdited);
        rafinazariMiddleRadius->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariMiddleRadius ? Edited : UnEdited);
        rafinazariFarRadius->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariFarRadius ? Edited : UnEdited);
        rafinazariEnergySigma->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariEnergySigma ? Edited : UnEdited);
        rafinazariEnergyRadius->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariEnergyRadius ? Edited : UnEdited);
        rafinazariEnergyBoxRadius->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariEnergyBoxRadius ? Edited : UnEdited);
        rafinazariEnergyFloor->setDefaultEditedState(pedited->raw.xtranssensor.rafinazariEnergyFloor ? Edited : UnEdited);
    } else {
        dualDemosaicContrast->setDefaultEditedState(Irrelevant );
        border->setDefaultEditedState(Irrelevant);
        ccSteps->setDefaultEditedState(Irrelevant );
        rafinazariSigma->setDefaultEditedState(Irrelevant);
        rafinazariNearRadius->setDefaultEditedState(Irrelevant);
        rafinazariMiddleRadius->setDefaultEditedState(Irrelevant);
        rafinazariFarRadius->setDefaultEditedState(Irrelevant);
        rafinazariEnergySigma->setDefaultEditedState(Irrelevant);
        rafinazariEnergyRadius->setDefaultEditedState(Irrelevant);
        rafinazariEnergyBoxRadius->setDefaultEditedState(Irrelevant);
        rafinazariEnergyFloor->setDefaultEditedState(Irrelevant);
    }
}

void XTransProcess::adjusterChanged(Adjuster* a, double newval)
{
    if (listener) {
        if (a == ccSteps) {
            listener->panelChanged (EvDemosaicFalseColorIter, a->getTextValue() );
        } else if (a == dualDemosaicContrast) {
            listener->panelChanged (EvDemosaicContrast, a->getTextValue() );
        } else if (a == border) {
            listener->panelChanged (EvDemosaicBorder, a->getTextValue() );
        } else {
            listener->panelChanged(EvRafinazariSettings, a->getTextValue());
        }
    }
}

void XTransProcess::adjusterAutoToggled(Adjuster* a, bool newval)
{
    if (multiImage) {
        if (dualDemosaicContrast->getAutoInconsistent()) {
            dualDemosaicContrast->setAutoInconsistent (false);
            dualDemosaicContrast->setAutoValue (false);
        } else if (lastAutoContrast) {
            dualDemosaicContrast->setAutoInconsistent (true);
        }

        lastAutoContrast = dualDemosaicContrast->getAutoValue();
    }

    if (listener) {

        if (a == dualDemosaicContrast) {
            if (dualDemosaicContrast->getAutoInconsistent()) {
                listener->panelChanged (EvDemosaicAutoContrast, M ("GENERAL_UNCHANGED"));
            } else if (dualDemosaicContrast->getAutoValue()) {
                listener->panelChanged (EvDemosaicAutoContrast, M ("GENERAL_ENABLED"));
            } else {
                listener->panelChanged (EvDemosaicAutoContrast, M ("GENERAL_DISABLED"));
            }
        }
    }
}

void XTransProcess::methodChanged ()
{
    const int curSelection = method->get_active_row_number();
    const RAWParams::XTransSensor::Method currentMethod = RAWParams::XTransSensor::Method(curSelection);

    oldSelection = curSelection;

    if (!batchMode) {
        if (currentMethod == procparams::RAWParams::XTransSensor::Method::FOUR_PASS || currentMethod == procparams::RAWParams::XTransSensor::Method::TWO_PASS) {
            dualDemosaicOptions->show();
        } else {
            dualDemosaicOptions->hide();
        }
        rafinazariOptions->set_visible(currentMethod == procparams::RAWParams::XTransSensor::Method::RAFINAZARI);

    }
    if (listener && method->get_active_row_number() >= 0) {
        listener->panelChanged (
            currentMethod == RAWParams::XTransSensor::Method::MONO || RAWParams::XTransSensor::Method(oldSelection) == RAWParams::XTransSensor::Method::MONO
            ? EvDemosaicMethodPreProc
            : EvDemosaicMethod, method->get_active_text());
    }
}

void XTransProcess::checkBoxToggled (CheckBox* c, CheckValue newval)
{
}

void XTransProcess::autoContrastChanged (double autoContrast)
{
    idle_register.add(
        [this, autoContrast]() -> bool
        {
            disableListener();
            dualDemosaicContrast->setValue(autoContrast);
            enableListener();
            return false;
        }
    );
}
