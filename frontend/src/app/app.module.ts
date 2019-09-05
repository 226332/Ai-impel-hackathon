import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { BackgroundComponent } from './background/background.component';
import { PanelComponent } from './panel/panel.component';
import { NgxFileDropModule } from 'ngx-file-drop';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import { FireComponent } from './fire/fire.component';
import {HttpClientModule} from '@angular/common/http';
@NgModule({
  declarations: [
    AppComponent,
    BackgroundComponent,
    PanelComponent,
    FireComponent
  ],
  imports: [
    NgxFileDropModule,
    BrowserAnimationsModule,
    HttpClientModule,
    BrowserModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
