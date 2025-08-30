#!/bin/sh
# ----------------------------------------------------------------------------
# Apache Maven Wrapper
# ----------------------------------------------------------------------------

# Check if Java is available
if [ -z "$JAVA_HOME" ]; then
  JAVA_EXECUTABLE="java"
else
  JAVA_EXECUTABLE="$JAVA_HOME/bin/java"
fi

# Maven wrapper JAR location
MAVEN_WRAPPER_JAR=".mvn/wrapper/maven-wrapper.jar"

# Check if wrapper JAR exists
if [ ! -f "$MAVEN_WRAPPER_JAR" ]; then
  echo "Maven wrapper JAR not found at $MAVEN_WRAPPER_JAR"
  echo "Please run: curl -s https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.2.0/maven-wrapper-3.2.0.jar -o .mvn/wrapper/maven-wrapper.jar"
  exit 1
fi

# Execute Maven wrapper
exec "$JAVA_EXECUTABLE" \
  $MAVEN_OPTS \
  -classpath "$MAVEN_WRAPPER_JAR" \
  "-Dmaven.multiModuleProjectDirectory=." \
  org.apache.maven.wrapper.MavenWrapperMain "$@"